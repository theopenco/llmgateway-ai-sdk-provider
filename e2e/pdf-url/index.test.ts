import type { ModelMessage } from 'ai';

import { createLLMGateway } from '@/src';
import { generateText } from 'ai';
import { expect, test, vi } from 'vitest';

vi.setConfig({
  testTimeout: 42_000,
});

test('send pdf urls', async () => {
  const llmgateway = createLLMGateway({
    apiKey: process.env.LLM_GATEWAY_API_KEY,
    baseUrl: process.env.LLM_GATEWAY_API_BASE,
  });

  const model = llmgateway('gpt-4o', {
    usage: {
      include: true,
    },
  });
  const messageHistory: ModelMessage[] = [];
  messageHistory.push({
    role: 'user',
    content: [
      {
        type: 'text',
        text: "What's in this file?",
      },
      {
        type: 'file',
        data: new URL('https://bitcoin.org/bitcoin.pdf'),
        mediaType: 'application/pdf',
      },
    ],
  });

  const response = await generateText({
    model,
    messages: messageHistory,
  });

  expect(response.text).toMatch(/bitcoin|electronic cash/i);
});
