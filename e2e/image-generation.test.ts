import { createLLMGateway } from '@/src';
import { generateImage, generateText, streamText } from 'ai';
import { expect, it, vi } from 'vitest';

vi.setConfig({ testTimeout: 120_000 });

const provider = createLLMGateway({
  apiKey: process.env.LLM_GATEWAY_API_KEY,
  baseURL: process.env.LLM_GATEWAY_API_BASE,
});

it('generates an image through generateImage', async () => {
  const result = await generateImage({
    model: provider.image(
      process.env.LLM_GATEWAY_IMAGE_MODEL ?? 'qwen-image-plus',
    ),
    prompt: 'A small orange cat sitting on a plain white background',
  });
  expect(result.images).toHaveLength(1);
  expect(result.image.mediaType).toMatch(/^image\//);
  expect(result.image.uint8Array.byteLength).toBeGreaterThan(100);
});

it.each([false, true])(
  'returns generated chat images (streaming: %s)',
  async (streaming) => {
    const options = {
      model: provider(
        process.env.LLM_GATEWAY_CHAT_IMAGE_MODEL ?? 'gemini-2.5-flash-image',
      ),
      prompt:
        'Generate an image of a small orange cat on a plain white background.',
    };
    if (streaming) {
      const result = streamText(options);
      await result.consumeStream();
      const files = await result.files;
      expect(files.length).toBeGreaterThan(0);
      expect(files[0]?.mediaType).toMatch(/^image\//);
      expect(files[0]?.uint8Array.byteLength).toBeGreaterThan(100);
    } else {
      const result = await generateText(options);
      expect(result.files.length).toBeGreaterThan(0);
      expect(result.files[0]?.mediaType).toMatch(/^image\//);
      expect(result.files[0]?.uint8Array.byteLength).toBeGreaterThan(100);
    }
  },
);
