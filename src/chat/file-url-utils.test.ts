import { getFileUrl } from './file-url-utils';

describe('v4 file data', () => {
  it('encodes inline Unicode text as UTF-8', () => {
    expect(
      getFileUrl({
        part: {
          type: 'file',
          mediaType: 'text/plain',
          data: { type: 'text', text: 'Hej 🌊' },
        },
        defaultMediaType: 'application/pdf',
      }),
    ).toBe('data:text/plain;base64,SGVqIPCfjIo=');
  });

  it('rejects provider file references that the gateway cannot resolve', () => {
    expect(() =>
      getFileUrl({
        part: {
          type: 'file',
          mediaType: 'application/pdf',
          data: { type: 'reference', reference: { openai: 'file_123' } },
        },
        defaultMediaType: 'application/pdf',
      }),
    ).toThrow('provider file references');
  });
});
