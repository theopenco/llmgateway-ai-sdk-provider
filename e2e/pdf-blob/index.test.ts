import type { ModelMessage } from 'ai';

import { createLLMGateway } from '@/src';
import { generateText } from 'ai';
import { expect, test, vi } from 'vitest';

vi.setConfig({
  testTimeout: 42_000,
});

test('sending pdf base64 blob', async () => {
  const llmgateway = createLLMGateway({
    apiKey: process.env.LLM_GATEWAY_API_KEY,
    baseUrl: process.env.LLM_GATEWAY_API_BASE,
  });

  const model = llmgateway('gpt-4o', {
    usage: {
      include: true,
    },
  });

  const pdfBlob = await fetch('https://bitcoin.org/bitcoin.pdf').then((res) =>
    res.arrayBuffer(),
  );

  const pdfBase64 = Buffer.from(pdfBlob).toString('base64');

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
        data: `data:application/pdf;base64,${pdfBase64}`,
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
