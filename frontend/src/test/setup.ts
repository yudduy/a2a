import '@testing-library/jest-dom'
import { vi, expect, test, describe, beforeEach, afterEach, beforeAll, afterAll } from 'vitest'

// Make vitest globals available - cast to any to avoid TypeScript errors
(global as any).vi = vi;
(global as any).expect = expect;
(global as any).test = test;
(global as any).describe = describe;
(global as any).beforeEach = beforeEach;
(global as any).afterEach = afterEach;
(global as any).beforeAll = beforeAll;
(global as any).afterAll = afterAll;

// Mock ResizeObserver
global.ResizeObserver = class ResizeObserver {
  constructor(cb: any) {}
  observe() {}
  unobserve() {}
  disconnect() {}
};

// Mock IntersectionObserver
global.IntersectionObserver = class IntersectionObserver {
  constructor(cb: any, options?: any) {}
  observe() {}
  unobserve() {}
  disconnect() {}
  root = null;
  rootMargin = '';
  thresholds = [];
} as any;

// Mock window.matchMedia
Object.defineProperty(window, 'matchMedia', {
  writable: true,
  value: (query: string) => ({
    matches: false,
    media: query,
    onchange: null,
    addListener: () => {},
    removeListener: () => {},
    addEventListener: () => {},
    removeEventListener: () => {},
    dispatchEvent: () => {},
  }),
});

// Mock requestAnimationFrame
global.requestAnimationFrame = (cb: any) => setTimeout(cb, 0) as any;
global.cancelAnimationFrame = (id: any) => clearTimeout(id);

// Increase test timeout for typing animations
vi.setConfig({ testTimeout: 10000 });