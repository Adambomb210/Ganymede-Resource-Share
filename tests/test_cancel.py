"""The cancel transport and the image handles on a task payload (docs/11 §3, §4).

§3's whole delivery mechanism is one field on a heartbeat response. That makes
it cheap to get subtly wrong in a way nothing notices: a cancel that never
reaches the worker looks exactly like a worker that has not heartbeated yet, and
a job cancelled by an operator that lands on the *contributor's* record as an
abandonment is a quiet unfairness nobody would go looking for. Both are asserted
here.

The worker-side half of §3 -- what a worker actually does to a job container on
seeing the field -- is in test_sandbox.py; this is the coordinator's end.
"""

from __future__ import annotations

import json
import uuid
from datetime import timedelta

import pytest

from ganymede.coordinator import ledger, rounds
from ganymede.coordinator.store import image_key
from tests.fake_worker import FakeWorker
from tests.test_images import VETTED, docker_archive


def _hdr(key: str) -> dict:
    return {"Authorization": f"Bearer {key}"}


@pytest.fixture
def make_submitter(conn, make_contributor):
    def _make(name: str = "submitter"):
        cid, key = make_contributor(name=name)
        conn.execute(
            "INSERT INTO submitters (user_id, status, decided_at) VALUES (?, ?, ?)",
            (cid, "approved", rounds._iso(rounds.utcnow())),
        )
        conn.commit()
        return cid, key
    return _make


def _job_id(conn, run_id: str) -> str:
    return conn.execute(
        "SELECT job_id FROM runs WHERE id = ?", (run_id,)
    ).fetchone()["job_id"]


def _cancel(client, conn, run_id: str, mode: str, key: str) -> None:
    job_id = _job_id(conn, run_id)
    # seeded_run's job is owned by the synthetic `system` contributor, so the
    # caller here is an admin -- which is also the realistic path for a run an
    # operator seeded rather than submitted.
    r = client.post(f"/v1/jobs/{job_id}/cancel", headers=_hdr(key),
                    json={"mode": mode})
    assert r.status_code == 200, r.text


@pytest.fixture
def admin(conn):
    from ganymede.coordinator.auth import generate_key, hash_key

    cid, key = uuid.uuid4().hex, generate_key()
    conn.execute(
        """INSERT INTO contributors
             (id, name, key_hash, enabled, clearance, is_admin, created_at)
           VALUES (?, ?, ?, 1, 'open', 1, ?)""",
        (cid, "admin", hash_key(key), rounds._iso(rounds.utcnow())),
    )
    conn.commit()
    return cid, key


# ==========================================================================
# The transport (docs/11 §3 steps 1-3)
# ==========================================================================


def test_a_heartbeat_carries_no_cancel_field_normally(client, store,
                                                      make_contributor, seeded_run):
    """The field is absent, not false. A worker that has never seen a cancel
    should not have to distinguish two shapes of 'no'."""
    seeded_run()
    _, key = make_contributor()
    fw = FakeWorker(client, store, key)
    task = fw.claim()
    body = fw.heartbeat(1)
    assert "cancel" not in body
    assert task is not None


@pytest.mark.parametrize("mode", ["soft", "hard"])
def test_the_cancel_rides_the_next_heartbeat(client, conn, store, admin,
                                             make_contributor, seeded_run, mode):
    seeded_run()
    _, key = make_contributor()
    fw = FakeWorker(client, store, key)
    fw.claim()

    _cancel(client, conn, "run1", mode, admin[1])
    assert fw.heartbeat(5)["cancel"] == mode


def test_a_cancel_without_a_mode_reads_as_soft(client, conn, store,
                                               make_contributor, seeded_run):
    """`hard` is never something to infer: a job row that somehow carries a
    cancelled status and no mode gets the gentler reading."""
    seeded_run()
    _, key = make_contributor()
    fw = FakeWorker(client, store, key)
    fw.claim()
    conn.execute(
        "UPDATE jobs SET status = 'cancelled', cancel_mode = NULL WHERE id = ?",
        (_job_id(conn, "run1"),),
    )
    conn.commit()
    assert fw.heartbeat(1)["cancel"] == "soft"


def test_only_this_jobs_tasks_hear_about_it(client, conn, store, admin,
                                            make_contributor, seeded_run):
    seeded_run(run_id="doomed")
    seeded_run(run_id="fine")
    _, key_a = make_contributor("a")
    _, key_b = make_contributor("b")
    fw_a = FakeWorker(client, store, key_a)
    fw_b = FakeWorker(client, store, key_b, device="RTX 4090")
    task_a = fw_a.claim(run_id="doomed")
    task_b = fw_b.claim(run_id="fine")
    assert task_a is not None and task_b is not None

    _cancel(client, conn, "doomed", "hard", admin[1])
    assert fw_a.heartbeat(1)["cancel"] == "hard"
    assert "cancel" not in fw_b.heartbeat(1)


# ==========================================================================
# What the release lands on (docs/11 §3 steps 4-5)
# ==========================================================================


def test_abandon_under_cancel_is_cancelled_not_abandoned(client, conn, store,
                                                         admin, make_contributor,
                                                         seeded_run):
    seeded_run()
    _, key = make_contributor()
    fw = FakeWorker(client, store, key)
    task = fw.claim()
    _cancel(client, conn, "run1", "soft", admin[1])
    fw.abandon()

    status = conn.execute("SELECT status FROM tasks WHERE id = ?",
                          (task["task_id"],)).fetchone()["status"]
    assert status == "cancelled"


def test_an_ordinary_abandon_is_still_abandoned(client, conn, store,
                                                make_contributor, seeded_run):
    seeded_run()
    _, key = make_contributor()
    fw = FakeWorker(client, store, key)
    task = fw.claim()
    fw.abandon()
    status = conn.execute("SELECT status FROM tasks WHERE id = ?",
                          (task["task_id"],)).fetchone()["status"]
    assert status == "abandoned"


def test_a_cancelled_task_is_not_counted_against_the_machine(
        client, conn, store, admin, make_contributor, seeded_run):
    """docs/09 5.2 counts `abandoned` and `expired` as the machine's
    infractions. An operator cancelling a job is not the contributor's fault,
    and the separate status is what keeps it off their record."""
    seeded_run()
    _, key = make_contributor()
    fw = FakeWorker(client, store, key)
    fw.claim()
    _cancel(client, conn, "run1", "soft", admin[1])
    fw.abandon()

    since = rounds._iso(rounds.utcnow() - timedelta(days=1))
    assert ledger._infraction_since(conn, fw.worker_id, since) is False


def test_an_ordinary_abandon_is_counted(client, conn, store, make_contributor,
                                        seeded_run):
    """The other half: the distinction only means something if the ordinary
    case still lands on the record."""
    seeded_run()
    _, key = make_contributor()
    fw = FakeWorker(client, store, key)
    fw.claim()
    fw.abandon()
    since = rounds._iso(rounds.utcnow() - timedelta(days=1))
    assert ledger._infraction_since(conn, fw.worker_id, since) is True


def test_a_wedged_worker_expires_onto_cancelled(client, conn, store, admin,
                                                make_contributor, seeded_run):
    """The wedged-worker path (docs/11 §3, second half): nobody drained
    anything, the soft/hard distinction collapsed to hard, and the coordinator
    reclaims the lease on its own. It still must not read as an infraction."""
    seeded_run()
    _, key = make_contributor()
    fw = FakeWorker(client, store, key)
    task = fw.claim()
    _cancel(client, conn, "run1", "hard", admin[1])

    conn.execute("UPDATE tasks SET lease_expires_at = ? WHERE id = ?",
                 (rounds._iso(rounds.utcnow() - timedelta(hours=1)), task["task_id"]))
    conn.commit()
    rounds.expire_leases(conn)

    status = conn.execute("SELECT status FROM tasks WHERE id = ?",
                          (task["task_id"],)).fetchone()["status"]
    assert status == "cancelled"
    since = rounds._iso(rounds.utcnow() - timedelta(days=1))
    assert ledger._infraction_since(conn, fw.worker_id, since) is False


def test_an_ordinary_lease_still_expires(client, conn, store, make_contributor,
                                         seeded_run):
    seeded_run()
    _, key = make_contributor()
    fw = FakeWorker(client, store, key)
    task = fw.claim()
    conn.execute("UPDATE tasks SET lease_expires_at = ? WHERE id = ?",
                 (rounds._iso(rounds.utcnow() - timedelta(hours=1)), task["task_id"]))
    conn.commit()
    assert rounds.expire_leases(conn) == 1
    status = conn.execute("SELECT status FROM tasks WHERE id = ?",
                          (task["task_id"],)).fetchone()["status"]
    assert status == "expired"


def test_a_cancelled_shard_is_not_re_dispatched(client, conn, store, admin,
                                                make_contributor, seeded_run):
    """§3 step 5: `leased -> cancelled` is terminal, so no second container
    starts on a unit of work that is draining."""
    seeded_run()
    _, key = make_contributor()
    fw = FakeWorker(client, store, key)
    fw.claim()
    _cancel(client, conn, "run1", "soft", admin[1])
    fw.abandon()

    _, other_key = make_contributor("second")
    other = FakeWorker(client, store, other_key, device="RTX 4090")
    assert other.claim() is None


# ==========================================================================
# The image handles (docs/06 "Claim path", docs/11 §4)
# ==========================================================================


def test_a_built_in_task_carries_no_image_handles(client, store,
                                                  make_contributor, seeded_run):
    """docs/11 §4: a first-party type takes none of the §2 path, and the three
    nulls are how the worker knows that without a second flag."""
    seeded_run()
    _, key = make_contributor()
    task = FakeWorker(client, store, key).claim()
    assert task["image_ref"] is None
    assert task["image_digest"] is None
    assert task["image_pull_url"] is None


def test_a_submitter_task_carries_the_archive_and_its_digest(
        client, conn, store, make_submitter, make_contributor, seeded_run):
    """The digest is the one value the worker can recompute from the bytes it
    pulls (§2.3), and this row is the only place it can come from -- the
    coordinator never hashes the archive itself."""
    import hashlib

    from ganymede.coordinator import images

    _, sub_key = make_submitter()
    payload = docker_archive()
    r = client.post("/v1/images/upload-url", headers=_hdr(sub_key), json={
        "repo_tag": "job:latest", "digest": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload)})
    image_id = r.json()["image_id"]
    store.put_bytes(image_key(image_id), payload)
    client.post(f"/v1/images/{image_id}/finalize", headers=_hdr(sub_key))
    images.drain_pending(conn, store, images.ScanLimits(
        vetted_base_diff_ids=frozenset({VETTED})))

    seeded_run()
    conn.execute("UPDATE jobs SET image_id = ? WHERE id = ?",
                 (image_id, _job_id(conn, "run1")))
    conn.commit()

    _, key = make_contributor()
    task = FakeWorker(client, store, key, container_runtime="docker").claim()
    assert task is not None
    assert task["image_ref"] == image_id
    assert task["image_digest"] == hashlib.sha256(payload).hexdigest()
    assert image_key(image_id) in task["image_pull_url"]


def test_a_machine_with_no_container_runtime_is_refused_with_the_reason(
        client, conn, store, make_submitter, make_contributor, seeded_run):
    """docs/11 §4 names the string, so the scheduler doc and the sandbox doc
    agree on it. A machine that never opted into Docker never sees submitter
    code -- and the contributor can find out why."""
    import hashlib

    from ganymede.coordinator import eligibility, images

    _, sub_key = make_submitter()
    payload = docker_archive()
    r = client.post("/v1/images/upload-url", headers=_hdr(sub_key), json={
        "repo_tag": "job:latest", "digest": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload)})
    image_id = r.json()["image_id"]
    store.put_bytes(image_key(image_id), payload)
    client.post(f"/v1/images/{image_id}/finalize", headers=_hdr(sub_key))
    images.drain_pending(conn, store, images.ScanLimits(
        vetted_base_diff_ids=frozenset({VETTED})))

    seeded_run()
    job_id = _job_id(conn, "run1")
    conn.execute("UPDATE jobs SET image_id = ? WHERE id = ?", (image_id, job_id))
    conn.commit()

    _, key = make_contributor()
    fw = FakeWorker(client, store, key)  # no container_runtime
    assert fw.claim() is None

    verdicts = {v.job_id: v for v in eligibility.explain(conn, fw.worker_id).verdicts}
    assert verdicts[job_id].outcome == eligibility.REFUSED
    assert verdicts[job_id].reason == "no_container_runtime"
