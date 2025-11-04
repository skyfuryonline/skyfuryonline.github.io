/* ===========================================================
 * sw.js (Network First Strategy)
 * ===========================================================
 * This service worker prioritizes fetching fresh content from the
 * network. If the network is unavailable, it falls back to cache.
 * ========================================================== */

const CACHE_NAME = 'skyfury-blog-cache-v1';
const OFFLINE_URL = 'offline.html';

// On install, pre-cache the offline page
self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => {
      return cache.add(OFFLINE_URL);
    })
  );
  self.skipWaiting();
});

// On activate, clean up old caches
self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((cacheNames) => {
      return Promise.all(
        cacheNames.map((cacheName) => {
          if (cacheName !== CACHE_NAME) {
            console.log('Deleting old cache:', cacheName);
            return caches.delete(cacheName);
          }
        })
      );
    }).then(() => self.clients.claim())
  );
});

// On fetch, implement the Network First strategy
self.addEventListener('fetch', (event) => {
  // We only want to intercept navigation requests
  if (event.request.mode === 'navigate') {
    event.respondWith(
      fetch(event.request)
        .then((response) => {
          // If the fetch is successful, clone it and cache it
          const responseToCache = response.clone();
          caches.open(CACHE_NAME).then((cache) => {
            cache.put(event.request, responseToCache);
          });
          return response;
        })
        .catch(() => {
          // If the fetch fails (e.g., offline), try to get it from the cache
          return caches.match(event.request).then((response) => {
            // If it's in the cache, return it. Otherwise, show the offline page.
            return response || caches.match(OFFLINE_URL);
          });
        })
    );
  }
  // For non-navigation requests (CSS, JS, images), let them pass through.
  // The cache-busting query string we added earlier will handle them.
  return;
});