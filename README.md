# LLMGateway Provider for Vercel AI SDK

Forked from https://github.com/OpenRouterTeam/ai-sdk-provider

The [LLMGateway](https://llmgateway.io/) provider for the [Vercel AI SDK](https://ai-sdk.dev/docs) supports text, image, and video generation through LLMGateway.

# AI SDK version support

- For AI SDK v4 or lower, use `@llmgateway/ai-sdk-provider@1.x.x`.
- For AI SDK v5, use `@llmgateway/ai-sdk-provider@2.x.x`.
- For AI SDK v6, use `@llmgateway/ai-sdk-provider@3.x.x`.
- For AI SDK v7, use `@llmgateway/ai-sdk-provider@4.x.x`.

This version requires AI SDK **7.0.93 or later**, **Node.js 22 or later**, and ESM imports. It implements the SDK's v4 model interfaces, including native asynchronous video generation. Applications migrating from v6 should follow the [AI SDK 7 migration guide](https://ai-sdk.dev/docs/migration-guides/migration-guide-7-0).

## Setup

```bash
# For npm
npm install @llmgateway/ai-sdk-provider ai zod

# For pnpm
pnpm add @llmgateway/ai-sdk-provider ai zod

# For yarn
yarn add @llmgateway/ai-sdk-provider ai zod
```

## Provider Instance

You can import the default provider instance `llmgateway` from `@llmgateway/ai-sdk-provider`:

```ts
import { llmgateway } from '@llmgateway/ai-sdk-provider';
```

## Example

```ts
import { createLLMGateway } from '@llmgateway/ai-sdk-provider';
import { generateText } from 'ai';

const llmgateway = createLLMGateway({
  apiKey: process.env.LLM_GATEWAY_API_KEY,
});

const { text } = await generateText({
  model: llmgateway('openai/gpt-4o'),
  prompt: 'Write a vegetarian lasagna recipe for 4 people.',
});

console.log(`response: ${text}`);
```

## Image generation

```ts
import { llmgateway } from '@llmgateway/ai-sdk-provider';
import { generateImage } from 'ai';

const { image } = await generateImage({
  model: llmgateway.image('qwen-image-plus'),
  prompt: 'A watercolor painting of a mountain lake',
});

// image.uint8Array, image.base64, and image.mediaType contain the result.
```

## Video generation

Use `llmgateway.video(modelId)` or its alias `llmgateway.videoModel(modelId)` with the SDK's experimental video API. Model IDs are strings, so newly added gateway video models work without a provider update.

```ts
import { writeFile } from 'node:fs/promises';
import { llmgateway } from '@llmgateway/ai-sdk-provider';
import { experimental_generateVideo as generateVideo } from 'ai';

const { video } = await generateVideo({
  model: llmgateway.video('veo-3.1-fast-generate-preview'),
  prompt: 'A cinematic aerial view of ocean waves at sunrise',
  duration: 8,
  resolution: '1280x720',
  poll: { intervalMs: 5_000, timeoutMs: 600_000 },
  abortSignal: AbortSignal.timeout(600_000),
});

await writeFile('video.mp4', video.uint8Array);
```

The provider creates a job, lets the SDK poll its status, and downloads the video through the authenticated gateway content endpoint. Failed, canceled, and expired jobs produce errors. The result includes video bytes, media type, response metadata, and the job ID in `providerMetadata.llmgateway.videos`.

`duration` and a text prompt are required by LLMGateway. Choose a duration and resolution supported by your model; see the [video generation API](https://docs.llmgateway.io/features/video-generation). `resolution` maps to `size`, `duration` to `seconds`, and `generateAudio` to `audio`. `aspectRatio`, `fps`, and `seed` produce unsupported-setting warnings; use `resolution` to choose the aspect ratio. Use `n` to generate multiple videos as separate jobs.

For models that support image-to-video, supply an image URL, base64 data, or bytes:

```ts
const { video } = await generateVideo({
  model: llmgateway.video('seedance-2-0'),
  prompt: {
    image: 'https://example.com/first-frame.png',
    text: 'Animate the scene with gentle camera motion',
  },
  duration: 5,
  resolution: '1280x720',
  generateAudio: false,
});
```

The SDK's `frameImages` maps first and last frames to `image` and `last_frame`. `inputReferences` maps images and videos to `reference_images` and `reference_videos`; video references must be HTTPS URLs with a video media type. Model-specific support and input combinations are validated by the gateway.

Additional gateway fields, such as reference audio or signed callback settings, can be passed through `providerOptions.llmgateway`. You can also set `extraBody` on `createLLMGateway` or on a video model. Call options take precedence over model `extraBody`, then provider `extraBody`. The SDK's `poll` settings control polling and are not sent to the gateway. Aborting stops local requests and polling; the gateway job may continue running.

To persist a job and check it from another process, use the same model with the SDK's `experimental_startVideo` and `experimental_getVideoStatus` APIs. SDK-managed webhook completion is not advertised; callbacks can be configured directly using the gateway's `callback_url` and `callback_secret` fields.

## Testing

`pnpm test` runs the unit and SDK integration suites in Node and Edge environments. `pnpm typecheck`, `pnpm lint`, `pnpm stylecheck`, and `pnpm build` check the package and its public declarations. After building, `pnpm test:package` verifies the ESM exports and video generation through the built package.

For live API tests, copy `example.env.e2e` to `.env.e2e`, set `LLM_GATEWAY_API_KEY`, and run `pnpm test:e2e`. These tests submit real requests, including image and video generation, and consume gateway credits. Optional model overrides are listed in the example environment file.

## Supported models

You can find the latest list of models supported by LLMGateway [here](https://llmgateway.io/models).
