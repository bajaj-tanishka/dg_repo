/**
 * Browser Web Speech API service:
 * - SpeechSynthesis for German (de-DE) & English (en-US) text-to-speech.
 * - SpeechRecognition (webkitSpeechRecognition) for German voice dictation with typed fallback.
 */

// Extend window for Webkit SpeechRecognition
declare global {
  interface Window {
    SpeechRecognition?: any;
    webkitSpeechRecognition?: any;
  }
}

export const speechService = {
  isTTSAvailable(): boolean {
    return typeof window !== 'undefined' && 'speechSynthesis' in window;
  },

  isSTTAvailable(): boolean {
    return (
      typeof window !== 'undefined' &&
      ('SpeechRecognition' in window || 'webkitSpeechRecognition' in window)
    );
  },

  getVoices(): Promise<SpeechSynthesisVoice[]> {
    return new Promise((resolve) => {
      if (!this.isTTSAvailable()) {
        resolve([]);
        return;
      }

      let voices = window.speechSynthesis.getVoices();
      if (voices.length > 0) {
        resolve(voices);
        return;
      }

      window.speechSynthesis.onvoiceschanged = () => {
        voices = window.speechSynthesis.getVoices();
        resolve(voices);
      };

      setTimeout(() => {
        resolve(window.speechSynthesis.getVoices());
      }, 500);
    });
  },

  async speakGerman(text: string): Promise<void> {
    if (!this.isTTSAvailable() || !text.trim()) return;

    this.stop();
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = 'de-DE';
    utterance.rate = 0.95;

    const voices = await this.getVoices();
    const germanVoice = voices.find((v) => v.lang.startsWith('de') || v.lang.includes('DE'));
    if (germanVoice) {
      utterance.voice = germanVoice;
    }

    window.speechSynthesis.speak(utterance);
  },

  async speakEnglish(text: string): Promise<void> {
    if (!this.isTTSAvailable() || !text.trim()) return;

    this.stop();
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = 'en-US';
    utterance.rate = 1.0;

    const voices = await this.getVoices();
    const englishVoice = voices.find((v) => v.lang.startsWith('en') || v.lang.includes('US') || v.lang.includes('GB'));
    if (englishVoice) {
      utterance.voice = englishVoice;
    }

    window.speechSynthesis.speak(utterance);
  },

  stop(): void {
    if (this.isTTSAvailable()) {
      window.speechSynthesis.cancel();
    }
  },

  /**
   * Starts browser speech recognition listening for German spoken input.
   * Calls onResult with transcribed text and onEnd when stopped.
   */
  startSpeechRecognition(
    onResult: (transcript: string) => void,
    onError: (error: string) => void,
    onEnd: () => void
  ): { stop: () => void } | null {
    if (!this.isSTTAvailable()) {
      onError('Speech recognition is not supported in this browser. Please use the keyboard text input.');
      return null;
    }

    const SpeechRec = window.SpeechRecognition || window.webkitSpeechRecognition;
    const recognition = new SpeechRec();
    recognition.lang = 'de-DE';
    recognition.continuous = false;
    recognition.interimResults = true;

    recognition.onresult = (event: any) => {
      let finalTranscript = '';
      for (let i = event.resultIndex; i < event.results.length; ++i) {
        if (event.results[i].isFinal) {
          finalTranscript += event.results[i][0].transcript;
        } else {
          finalTranscript += event.results[i][0].transcript;
        }
      }
      onResult(finalTranscript);
    };

    recognition.onerror = (event: any) => {
      console.warn('[SpeechRecognition] error:', event.error);
      onError(event.error === 'not-allowed' ? 'Microphone permission denied.' : `Speech error: ${event.error}`);
    };

    recognition.onend = () => {
      onEnd();
    };

    try {
      recognition.start();
      return {
        stop: () => recognition.stop(),
      };
    } catch (err: any) {
      onError(err.message || 'Could not start microphone.');
      return null;
    }
  },
};
