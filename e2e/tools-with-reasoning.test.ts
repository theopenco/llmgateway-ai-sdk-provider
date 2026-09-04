import type { ModelMessage } from 'ai';

import {
  executeCommandInTerminalTool,
  readSMSTool,
  sendSMSTool,
} from '@/e2e/tools';
import { createLLMGateway } from '@/src';
import { generateText } from 'ai';
import { expect, it, vi } from 'vitest';

vi.setConfig({
  testTimeout: 42_000,
});

const prompts = [
  'The flight to San Francisco is delayed by 2 hours. Send an SMS to 808-999-2345 with the flight delay information.',
  'Find out if the SMS was delivered properly.',
];

describe('Vercel AI SDK tools call with reasoning', () => {
  it('should work with reasoning content', async () => {
    const llmgateway = createLLMGateway({
      apiKey: process.env.LLM_GATEWAY_API_KEY,
      baseUrl: process.env.LLM_GATEWAY_API_BASE,
    });

    const model = llmgateway('o4-mini', {
      usage: {
        include: true,
      },
    });
    const messageHistory: ModelMessage[] = [];
    for (const prompt of prompts) {
      messageHistory.push({
        role: 'user',
        content: [
          {
            type: 'text',
            text: prompt,
          },
        ],
      });

      const response = await generateText({
        model,
        instructions:
          'You are an airline assistant. You can send and read SMS messages, and execute commands in the terminal.',
        messages: messageHistory,
        tools: {
          readSMS: readSMSTool,
          sendSMS: sendSMSTool,
          executeCommand: executeCommandInTerminalTool,
        },
        toolChoice: {
          type: 'tool',
          toolName: prompt === prompts[0] ? 'sendSMS' : 'readSMS',
        },
        providerOptions: {
          llmgateway: {
            reasoningText: {
              max_tokens: 2048,
            },
          },
        },
      });

      expect(response.toolCalls.length).toBeGreaterThan(0);
      expect(response.toolResults.length).toBeGreaterThan(0);
      messageHistory.push(...response.response.messages);
    }
  });
});
