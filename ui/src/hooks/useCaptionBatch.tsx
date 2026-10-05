'use client';
import { useEffect, useRef, useState } from 'react';
import { apiClient } from '@/utils/api';

// Module-level batcher: many cards mount at once when the virtualized grid scrolls;
// instead of N HTTP requests, queue paths and flush them as a single batch.
// Entries are keyed by extension + path so different caption extensions don't
// collide in the cache or the pending batch.
type Resolver = { resolve: (caption: string) => void; reject: (err: unknown) => void };
type Pending = { path: string; ext: string; resolvers: Resolver[] };
const pending = new Map<string, Pending>();
const cache = new Map<string, string>();
const revisions = new Map<string, number>();
const subscribers = new Map<string, Set<(caption: string) => void>>();
let flushTimer: ReturnType<typeof setTimeout> | null = null;
const FLUSH_DELAY_MS = 30;
const MAX_BATCH = 200;

function normExt(ext: string | undefined): string {
  return (ext || 'txt').replace(/^\.+/, '').trim() || 'txt';
}

function keyFor(path: string, ext: string): string {
  return `${ext}\n${path}`;
}

function nextRevision(key: string): number {
  const revision = (revisions.get(key) ?? 0) + 1;
  revisions.set(key, revision);
  return revision;
}

function scheduleFlush() {
  if (flushTimer) return;
  flushTimer = setTimeout(flush, FLUSH_DELAY_MS);
}

async function flush() {
  flushTimer = null;
  if (pending.size === 0) return;

  // Drain up to MAX_BATCH entries; if more arrived, reschedule.
  const keys: string[] = [];
  for (const key of pending.keys()) {
    keys.push(key);
    if (keys.length >= MAX_BATCH) break;
  }
  // A newer read or viewer save must win over a slower, older batch response.
  const drained = keys.map(k => ({ ...pending.get(k)!, revision: nextRevision(k) }));
  for (const k of keys) pending.delete(k);

  // Group by extension; each extension is a separate batch request.
  const byExt = new Map<string, (Pending & { revision: number })[]>();
  for (const entry of drained) {
    const group = byExt.get(entry.ext);
    if (group) group.push(entry);
    else byExt.set(entry.ext, [entry]);
  }

  await Promise.all(
    Array.from(byExt.entries()).map(async ([ext, entries]) => {
      const paths = entries.map(e => e.path);
      try {
        const res = await apiClient.post('/api/caption/getBatch', { imgPaths: paths, ext });
        const captions: Record<string, string> = res.data?.captions ?? {};
        for (const { path, ext: e, resolvers, revision } of entries) {
          const key = keyFor(path, e);
          if (revisions.get(key) === revision) {
            setCachedCaption(path, captions[path] ?? '', e);
          }
          const value = cache.get(key);
          if (value === undefined) {
            for (const r of resolvers) r.reject(new DOMException('Superseded', 'AbortError'));
            continue;
          }
          for (const r of resolvers) r.resolve(value);
        }
      } catch (err) {
        for (const { resolvers } of entries) {
          for (const r of resolvers) r.reject(err);
        }
      }
    }),
  );

  if (pending.size > 0) scheduleFlush();
}

function requestCaption(path: string, ext: string, signal?: AbortSignal): Promise<string> {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) {
      reject(new DOMException('Aborted', 'AbortError'));
      return;
    }
    const key = keyFor(path, ext);
    const resolver: Resolver = { resolve, reject };
    const entry = pending.get(key);
    if (entry) {
      entry.resolvers.push(resolver);
    } else {
      pending.set(key, { path, ext, resolvers: [resolver] });
    }
    if (signal) {
      const onAbort = () => {
        // Remove this resolver from the pending batch. If no other card is
        // still waiting on the same key, drop the entry entirely so the next
        // batch doesn't include it.
        const e = pending.get(key);
        if (e) {
          const idx = e.resolvers.indexOf(resolver);
          if (idx >= 0) e.resolvers.splice(idx, 1);
          if (e.resolvers.length === 0) pending.delete(key);
        }
        reject(new DOMException('Aborted', 'AbortError'));
      };
      signal.addEventListener('abort', onAbort, { once: true });
    }
    scheduleFlush();
  });
}

export function invalidateCaption(path: string, ext?: string) {
  const key = keyFor(path, normExt(ext));
  nextRevision(key);
  cache.delete(key);
}

export function setCachedCaption(path: string, caption: string, ext?: string) {
  const key = keyFor(path, normExt(ext));
  nextRevision(key);
  cache.set(key, caption);
  subscribers.get(key)?.forEach(notify => notify(caption));
}

// Fetches caption for a path, using the module-level batcher + cache.
// Revalidate on mount/visibility changes; `refreshKey` also requests a fresh read.
export default function useCaptionBatch(imgPath: string | null, refreshKey: number = 0, ext: string = 'txt') {
  const captionExt = normExt(ext);
  const [caption, setCaption] = useState<string>(() => (imgPath ? (cache.get(keyFor(imgPath, captionExt)) ?? '') : ''));
  const [isLoaded, setIsLoaded] = useState<boolean>(() => Boolean(imgPath && cache.has(keyFor(imgPath, captionExt))));
  const lastPathRef = useRef<string | null>(null);

  useEffect(() => {
    if (!imgPath) {
      setCaption('');
      setIsLoaded(false);
      return;
    }

    const key = keyFor(imgPath, captionExt);
    const cached = cache.get(key);
    if (refreshKey > 0) invalidateCaption(imgPath, captionExt);
    const notify = (value: string) => {
      setCaption(value);
      setIsLoaded(true);
    };
    const listeners = subscribers.get(key) ?? new Set<(caption: string) => void>();
    listeners.add(notify);
    subscribers.set(key, listeners);
    if (cached !== undefined) {
      setCaption(cached);
      setIsLoaded(true);
    } else {
      setIsLoaded(false);
    }

    let cancelled = false;
    const controller = new AbortController();
    lastPathRef.current = imgPath;
    requestCaption(imgPath, captionExt, controller.signal)
      .then(value => {
        if (cancelled || lastPathRef.current !== imgPath) return;
        setCaption(value);
        setIsLoaded(true);
      })
      .catch(err => {
        if (err?.name === 'AbortError' || cancelled) return;
        console.error('Error fetching caption:', err);
        setIsLoaded(true);
      });

    return () => {
      cancelled = true;
      listeners.delete(notify);
      if (listeners.size === 0) subscribers.delete(key);
      controller.abort();
    };
  }, [imgPath, refreshKey, captionExt]);

  return { caption, isLoaded, setCaption };
}
