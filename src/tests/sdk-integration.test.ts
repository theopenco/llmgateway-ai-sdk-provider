import type { ModelMessage } from 'ai';

import {
  createProviderRegistry,
  generateImage,
  generateText,
  isStepCount,
  Output,
  streamText,
  tool,
} from 'ai';
import { vi } from 'vitest';
import { z } from 'zod/v4';

import { createLLMGateway } from '../index';
import { createTestServer } from './create-test-server';

const baseURL = 'https://test.llmgateway.io/v1';
const chatURL = `${baseURL}/chat/completions`;
const completionURL = `${baseURL}/completions`;
const imageURL = `${baseURL}/images/generations`;
const png =
  'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==';
const usage = { prompt_tokens: 12, completion_tokens: 4, total_tokens: 16 };

function chatResponse(content: string) {
  return {
    choices: [
      { message: { role: 'assistant', content }, finish_reason: 'stop' },
    ],
    usage,
  };
}

function setup() {
  const server = createTestServer({
    [chatURL]: {
      response: { type: 'json-value', body: chatResponse('Hello!') },
    },
    [completionURL]: {
      response: {
        type: 'json-value',
        body: { choices: [{ text: 'Hello!', finish_reason: 'stop' }], usage },
      },
    },
    [imageURL]: {
      response: { type: 'json-value', body: { data: [{ b64_json: png }] } },
    },
  });
  return {
    server,
    provider: createLLMGateway({
      apiKey: 'test',
      baseURL,
      fetch: server.fetch,
    }),
  };
}

function streamResponse(parts: unknown[]) {
  return {
    type: 'stream-chunks' as const,
    chunks: [
      ...parts.map((part) => `data: ${JSON.stringify(part)}\n\n`),
      'data: [DONE]\n\n',
    ],
  };
}

describe('AI SDK 7 integration', () => {
  it('works in an SDK provider registry and reports unsupported embeddings', () => {
    const { provider } = setup();
    const registry = createProviderRegistry({ llmgateway: provider });
    expect(registry.languageModel('llmgateway:some-model').modelId).toBe(
      'some-model',
    );
    expect(registry.imageModel('llmgateway:some-image-model').modelId).toBe(
      'some-image-model',
    );
    expect(() => registry.embeddingModel('llmgateway:unsupported')).toThrow(
      'No such embeddingModel',
    );
  });
  it.each(['chat', 'completion'] as const)(
    'generates text and usage using %s models',
    async (kind) => {
      const { provider } = setup();
      const model = provider[kind]('test-model');
      expect(model.specificationVersion).toBe('v4');
      const result = await generateText({
        model,
        instructions: 'Be concise.',
        prompt: 'Hello',
      });
      expect(result.text).toBe('Hello!');
      expect(result.finishReason).toBe('stop');
      expect(result.usage).toMatchObject({
        inputTokens: 12,
        outputTokens: 4,
        totalTokens: 16,
      });
    },
  );

  it.each(['chat', 'completion'] as const)(
    'streams text through the SDK using %s models',
    async (kind) => {
      const { provider, server } = setup();
      const url = kind === 'chat' ? chatURL : completionURL;
      server.urls[url]!.response = streamResponse([
        ...['Hello', '!'].map((text) => ({
          choices: [
            {
              ...(kind === 'chat' ? { delta: { content: text } } : { text }),
              finish_reason: null,
            },
          ],
        })),
        {
          choices: [
            {
              ...(kind === 'chat' ? { delta: {} } : { text: '' }),
              finish_reason: 'stop',
            },
          ],
          usage,
        },
      ]);
      const errors: unknown[] = [];
      const result = streamText({
        model: provider[kind]('test-model'),
        prompt: 'Hello',
        onError: (event) => {
          errors.push(event.error);
        },
      });
      const chunks: string[] = [];
      for await (const text of result.textStream) {
        chunks.push(text);
      }
      expect(errors).toEqual([]);
      expect(chunks.join('')).toBe('Hello!');
      expect(await result.text).toBe('Hello!');
      expect(await result.usage).toMatchObject({
        inputTokens: 12,
        outputTokens: 4,
      });
    },
  );

  it('generates structured output through Output.object', async () => {
    const { provider, server } = setup();
    server.urls[chatURL]!.response = {
      type: 'json-value',
      body: chatResponse('{"answer":42}'),
    };
    const result = await generateText({
      model: provider('test-model'),
      prompt: 'Answer with a number',
      output: Output.object({ schema: z.object({ answer: z.number() }) }),
    });
    expect(result.output).toEqual({ answer: 42 });
    expect(
      (await server.calls[0]!.requestBodyJson).response_format,
    ).toMatchObject({
      type: 'json_schema',
      json_schema: {
        strict: true,
        schema: { properties: { answer: { type: 'number' } } },
      },
    });
  });

  it('streams structured output through Output.object', async () => {
    const { provider, server } = setup();
    server.urls[chatURL]!.response = streamResponse([
      { choices: [{ delta: { content: '{"answer":' } }] },
      {
        choices: [{ delta: { content: '42}' }, finish_reason: 'stop' }],
        usage,
      },
    ]);
    const result = streamText({
      model: provider('test-model'),
      prompt: 'Answer with a number',
      output: Output.object({ schema: z.object({ answer: z.number() }) }),
    });
    await result.consumeStream();
    expect(await result.output).toEqual({ answer: 42 });
  });

  it('executes a tool and sends the tool result into the next step', async () => {
    const { provider, server } = setup();
    server.urls[chatURL]!.response = {
      type: 'json-value',
      body: {
        choices: [
          {
            message: {
              role: 'assistant',
              content: null,
              tool_calls: [
                {
                  id: 'call_1',
                  type: 'function',
                  function: {
                    name: 'weather',
                    arguments: '{"city":"Stockholm"}',
                  },
                },
              ],
            },
            finish_reason: 'tool_calls',
          },
        ],
        usage,
      },
    };
    const execute = vi.fn(async ({ city }: { city: string }) => {
      server.urls[chatURL]!.response = {
        type: 'json-value',
        body: chatResponse(`${city} is sunny.`),
      };
      return { forecast: 'sunny' };
    });
    const result = await generateText({
      model: provider('test-model'),
      prompt: 'Weather in Stockholm?',
      stopWhen: isStepCount(2),
      tools: {
        weather: tool({ inputSchema: z.object({ city: z.string() }), execute }),
      },
    });
    expect(execute).toHaveBeenCalledOnce();
    expect(result.steps).toHaveLength(2);
    expect(result.text).toBe('Stockholm is sunny.');
    expect((await server.calls[1]!.requestBodyJson).messages).toContainEqual({
      role: 'tool',
      tool_call_id: 'call_1',
      content: '{"forecast":"sunny"}',
    });
  });

  it('normalizes image, audio, and PDF bytes into the v4 file format', async () => {
    const { provider, server } = setup();
    const messages: ModelMessage[] = [
      {
        role: 'user',
        content: [
          { type: 'text', text: 'Describe these files' },
          {
            type: 'file',
            data: new Uint8Array([0, 1, 2]),
            mediaType: 'image/png',
          },
          {
            type: 'file',
            data: new Uint8Array([0, 1, 2]),
            mediaType: 'audio/wav',
          },
          {
            type: 'file',
            data: new Uint8Array([0, 1, 2]),
            mediaType: 'application/pdf',
            filename: 'test.pdf',
          },
        ],
      },
    ];
    await generateText({ model: provider('test-model'), messages });
    expect(
      (await server.calls[0]!.requestBodyJson).messages[0].content,
    ).toEqual([
      { type: 'text', text: 'Describe these files' },
      { type: 'image_url', image_url: { url: 'data:image/png;base64,AAEC' } },
      { type: 'input_audio', input_audio: { data: 'AAEC', format: 'wav' } },
      {
        type: 'file',
        file: {
          filename: 'test.pdf',
          file_data: 'data:application/pdf;base64,AAEC',
        },
      },
    ]);
  });

  it.each([false, true])(
    'returns chat-generated images through the SDK (streaming: %s)',
    async (streaming) => {
      const { provider, server } = setup();
      const content = {
        content: 'An image',
        images: [
          {
            type: 'image_url',
            image_url: { url: `data:image/png;base64,${png}` },
          },
        ],
      };
      server.urls[chatURL]!.response = streaming
        ? streamResponse([
            { choices: [{ delta: content, finish_reason: 'stop' }], usage },
          ])
        : {
            type: 'json-value',
            body: {
              choices: [
                {
                  message: { role: 'assistant', ...content },
                  finish_reason: 'stop',
                },
              ],
              usage,
            },
          };
      const options = { model: provider('image-model'), prompt: 'Draw a cat' };
      if (streaming) {
        const result = streamText(options);
        await result.consumeStream();
        const files = await result.files;
        expect(files[0]?.base64).toBe(png);
        expect(files[0]?.mediaType).toBe('image/png');
      } else {
        const result = await generateText(options);
        expect(result.files[0]?.base64).toBe(png);
        expect(result.files[0]?.mediaType).toBe('image/png');
      }
    },
  );

  it('generates images through the SDK', async () => {
    const { provider, server } = setup();
    const result = await generateImage({
      model: provider.image('qwen-image-plus'),
      prompt: 'Draw a cat',
      providerOptions: { llmgateway: { quality: 'high' } },
    });
    expect(result.image.base64).toBe(png);
    expect(result.image.mediaType).toBe('image/png');
    expect((await server.calls[0]!.requestBodyJson).quality).toBe('high');
  });
});
