/// <reference types="@raycast/api">

/* 🚧 🚧 🚧
 * This file is auto-generated from the extension's manifest.
 * Do not modify manually. Instead, update the `package.json` file.
 * 🚧 🚧 🚧 */

/* eslint-disable @typescript-eslint/ban-types */

type ExtensionPreferences = {
  /** whisper-realtime Path - Path to the whisper-realtime project directory */
  "whisperRealtimePath": string,
  /** Whisper Model - The Whisper model to use for transcription */
  "model": "tiny" | "base" | "small" | "medium" | "large-v3" | "large-v3-turbo",
  /** Language - Language for transcription */
  "language": "ja" | "en" | "auto",
  /** Speaker Diarization - Enable speaker diarization (identifies different speakers) */
  "enableSpeaker": boolean,
  /** Output Directory - Directory to save transcription and recording files */
  "outputDirectory": string,
  /** Audio Recording - Record audio while transcribing (saves as WAV file) */
  "enableRecording": boolean,
  /** Audio Source - Audio input source for transcription */
  "audioSource": "mic" | "system" | "both",
  /** Processing Step (ms) - Processing step interval in milliseconds (default: 500) */
  "processingStep": string,
  /** Processing Window (ms) - Processing window length in milliseconds - shorter means faster response (default: 3000) */
  "processingLength": string,
  /** Voice Activity Detection - Use VAD to filter out non-speech audio */
  "enableVad": boolean
}

/** Preferences accessible in all the extension's commands */
declare type Preferences = ExtensionPreferences

declare namespace Preferences {
  /** Preferences accessible in the `start-transcription` command */
  export type StartTranscription = ExtensionPreferences & {}
  /** Preferences accessible in the `voice-input` command */
  export type VoiceInput = ExtensionPreferences & {}
  /** Preferences accessible in the `manage-dictionary` command */
  export type ManageDictionary = ExtensionPreferences & {}
}

declare namespace Arguments {
  /** Arguments passed to the `start-transcription` command */
  export type StartTranscription = {}
  /** Arguments passed to the `voice-input` command */
  export type VoiceInput = {}
  /** Arguments passed to the `manage-dictionary` command */
  export type ManageDictionary = {}
}

