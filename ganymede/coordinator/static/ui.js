// Ganymede's own htmx wiring. Vendored rather than inline: `script-src 'self'`
// (docs/12) allows a served file and nothing else.
//
// htmx swaps 2xx responses only. Everything else is logged as a "Response
// Status Error Code" and the DOM is left untouched -- which silently defeated
// the one place this UI answers with 4xx *content* on purpose: a `/ui` form
// that re-renders itself carrying the reason it refused. Typing a malformed
// spec and pressing the button did nothing at all.
//
// So 422 from a `/ui/` endpoint is opted back in, and nothing else is. A 422
// from `/v1` is a FastAPI validation blob, and swapping that into the page is
// exactly what `/ui/jobs/new` exists to avoid.
document.addEventListener('htmx:beforeSwap', function (evt) {
  var xhr = evt.detail.xhr;
  var path = evt.detail.pathInfo ? evt.detail.pathInfo.requestPath : '';
  if (xhr && xhr.status === 422 && path && path.indexOf('/ui/') === 0) {
    evt.detail.shouldSwap = true;
    evt.detail.isError = false;
  }
});
