import type { LanguageModelV4FilePart } from '@ai-sdk/provider';

import { UnsupportedFunctionalityError } from '@ai-sdk/provider';
import { convertUint8ArrayToBase64 } from '@ai-sdk/provider-utils';

export function getFileUrl({
  part,
  defaultMediaType,
}: {
  part: LanguageModelV4FilePart;
  defaultMediaType: string;
}) {
  const mediaType = part.mediaType ?? defaultMediaType;
  switch (part.data.type) {
    case 'url':
      return part.data.url.toString();
    case 'data': {
      const base64 =
        typeof part.data.data === 'string'
          ? part.data.data
          : convertUint8ArrayToBase64(part.data.data);
      return `data:${mediaType};base64,${base64}`;
    }
    case 'text':
      return `data:${mediaType};base64,${convertUint8ArrayToBase64(
        new TextEncoder().encode(part.data.text),
      )}`;
    case 'reference':
      throw new UnsupportedFunctionalityError({
        functionality: 'provider file references',
      });
  }
}

export function getMediaType(
  dataUrl: string,
  defaultMediaType: string,
): string {
  const match = dataUrl.match(/^data:([^;]+)/);
  return match ? (match[1] ?? defaultMediaType) : defaultMediaType;
}

export function getBase64FromDataUrl(dataUrl: string): string {
  const match = dataUrl.match(/^data:[^;]*;base64,(.+)$/);
  return match ? match[1]! : dataUrl;
}
