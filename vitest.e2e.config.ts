import { config } from 'dotenv';
import { defineConfig } from 'vitest/config';

const envPath = new URL('./.env.e2e', import.meta.url);

// https://vitejs.dev/config/
export default defineConfig({
  resolve: {
    tsconfigPaths: true,
  },
  test: {
    pool: 'threads',
    environment: 'node',
    globals: true,
    env: config({ path: envPath.pathname, quiet: true }).parsed,
    include: ['./e2e/**/*.test.ts'],
  },
});
