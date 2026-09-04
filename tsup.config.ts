import { defineConfig } from 'tsup';

export default defineConfig({
  entry: {
    index: 'src/index.ts',
    'internal/index': 'src/internal/index.ts',
  },
  format: ['esm'],
  splitting: true,
  dts: true,
  sourcemap: true,
  clean: true,
});
