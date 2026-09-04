import { createLLMGateway } from '@/src';
import { experimental_generateVideo as generateVideo } from 'ai';
import { expect, it } from 'vitest';

it('generates and downloads a video through LLMGateway', async () => {
  const provider = createLLMGateway({
    apiKey: process.env.LLM_GATEWAY_API_KEY,
    baseURL: process.env.LLM_GATEWAY_API_BASE,
  });
  const result = await generateVideo({
    model: provider.video(
      process.env.LLM_GATEWAY_VIDEO_MODEL ?? 'veo-3.1-fast-generate-preview',
    ),
    prompt:
      'A static wide shot of gentle ocean waves on a sunny day. No people or text.',
    duration: Number(process.env.LLM_GATEWAY_VIDEO_DURATION ?? 8),
    resolution: '1280x720',
    maxRetries: 0,
    abortSignal: AbortSignal.timeout(600_000),
  });

  expect(result.videos).toHaveLength(1);
  expect(result.video.mediaType).toMatch(/^video\//);
  expect(result.video.uint8Array.byteLength).toBeGreaterThan(1000);
  expect(result.providerMetadata?.llmgateway?.videos).toEqual([
    expect.objectContaining({ id: expect.any(String), status: 'completed' }),
  ]);
}, 610_000);
