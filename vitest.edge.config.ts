import { defineConfig } from 'vitest/config';

// https://vitejs.dev/config/
export default defineConfig({
  resolve: {
    tsconfigPaths: true,
  },
  test: {
    pool: 'threads',
    environment: 'edge-runtime',
    globals: true,
    include: ['./src/**/*.test.ts'],
  },
});
