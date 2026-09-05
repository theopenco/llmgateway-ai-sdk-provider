import { createLLMGateway } from '@/src';
import { streamText } from 'ai';
import { expect, it } from 'vitest';

it('reuses a cached prompt', async () => {
  const provider = createLLMGateway({
    apiKey: process.env.LLM_GATEWAY_API_KEY,
    baseURL: process.env.LLM_GATEWAY_API_BASE,
    compatibility: 'strict',
  });
  const model = provider('anthropic/claude-sonnet-4-6');
  const call = async () => {
    const result = streamText({
      model,
      maxOutputTokens: 32,
      messages: [
        {
          role: 'user',
          content: [
            {
              type: 'text',
              text: 'This is a test document about ocean waves, weather, and coastal landscapes. '.repeat(
                600,
              ),
              providerOptions: {
                llmgateway: { cache_control: { type: 'ephemeral' } },
              },
            },
            {
              type: 'text',
              text: 'What is this document about? Reply in one sentence.',
            },
          ],
        },
      ],
    });
    await result.consumeStream();
    return result.usage;
  };
  await call();
  const usage = await call();
  expect(usage.inputTokenDetails.cacheReadTokens).toBeGreaterThan(0);
}, 120_000);
