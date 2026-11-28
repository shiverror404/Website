const V = 'jarvis-v11';
const SHELL = ['./', './index.html', './manifest.webmanifest', './icon-192.png', './icon-512.png', './logo.png'];
self.addEventListener('install', (e) => {
  e.waitUntil(caches.open(V).then((c) => c.addAll(SHELL)).then(() => self.skipWaiting()));
});
self.addEventListener('activate', (e) => {
  e.waitUntil(caches.keys().then((k) => Promise.all(k.filter((x) => x !== V).map((x) => caches.delete(x)))).then(() => self.clients.claim()));
});
// Sirf same-origin GET. Page: pehle network (taaki update turant dikhe), offline ho to cache.
// Baaki files: cache se turant, peeche se update. API calls ko haath nahi lagate.
self.addEventListener('fetch', (e) => {
  const r = e.request;
  if (r.method !== 'GET' || new URL(r.url).origin !== location.origin) return;
  const keep = (res) => { if (res.ok) { const cp = res.clone(); caches.open(V).then((c) => c.put(r, cp)); } return res; };
  if (r.mode === 'navigate') {
    e.respondWith(fetch(r).then(keep).catch(() => caches.match(r).then((h) => h || caches.match('./index.html'))));
    return;
  }
  e.respondWith(caches.match(r).then((hit) => {
    const net = fetch(r).then(keep).catch(() => hit);
    return hit || net;
  }));
});
