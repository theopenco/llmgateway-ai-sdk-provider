import { createLLMGateway } from '@/src';
import { streamText } from 'ai';
import { expect, it } from 'vitest';

it('receives standard and gateway usage accounting', async () => {
  const provider = createLLMGateway({
    apiKey: process.env.LLM_GATEWAY_API_KEY,
    baseURL: process.env.LLM_GATEWAY_API_BASE,
    compatibility: 'strict',
  });
  const result = streamText({
    model: provider('gpt-4o-mini', { usage: { include: true } }),
    prompt: 'What is the capital of France?',
    maxOutputTokens: 32,
  });
  await result.consumeStream();
  const usage = await result.usage;
  expect(usage.inputTokens).toBeGreaterThan(0);
  expect(usage.outputTokens).toBeGreaterThan(0);
  expect((await result.providerMetadata)?.llmgateway?.usage).toMatchObject({
    promptTokens: usage.inputTokens,
    completionTokens: usage.outputTokens,
    totalTokens: usage.totalTokens,
  });
}, 60_000);
