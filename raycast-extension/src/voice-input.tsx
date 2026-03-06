import {
  Action,
  ActionPanel,
  Detail,
  Icon,
  showToast,
  Toast,
  Clipboard,
  getPreferenceValues,
  closeMainWindow,
  popToRoot,
} from "@raycast/api";
import { useState, useEffect, useCallback, useRef } from "react";
import {
  getWhisperProcess,
  resetWhisperProcess,
  killAllWhisperProcesses,
  TranscriptionEntry,
  WhisperPreferences,
  StartOptions,
} from "./utils/whisper";
import { loadDictionary, applyDictionary } from "./utils/dictionary";

interface VoiceInputState {
  entries: TranscriptionEntry[]; // 文字起こしエントリ（isFinal情報を保持）
  isRecording: boolean;
  status: string;
  error: string | null;
  audioLevel: number;
  spectrum: number[];
  isProcessing: boolean; // 文字起こし処理中かどうか
  hasPartialText: boolean; // partialテキストがあるかどうか
}

export default function VoiceInput() {
  const [state, setState] = useState<VoiceInputState>({
    entries: [],
    isRecording: false,
    status: "Starting...",
    error: null,
    audioLevel: 0,
    spectrum: [0, 0, 0, 0, 0, 0, 0, 0],
    isProcessing: false,
    hasPartialText: false,
  });

  const preferences = getPreferenceValues<WhisperPreferences>();
  const hasStarted = useRef(false);

  // Shared startRecording function
  const startRecording = useCallback(async () => {
      try {
        if (!preferences.whisperRealtimePath) {
          setState((prev) => ({
            ...prev,
            error:
              "Please set the whisper-realtime path in extension preferences",
            status: "Error",
          }));
          await showToast({
            style: Toast.Style.Failure,
            title: "Configuration Required",
            message: "Set whisper-realtime path in preferences",
          });
          return;
        }

        // Kill any zombie processes first
        await killAllWhisperProcesses();

        // Reset any existing process for fresh voice input
        resetWhisperProcess();
        const process = getWhisperProcess();

        process.on("update", (entries: TranscriptionEntry[]) => {
          // 最後のentryがpartialかどうかをチェック
          const lastEntry = entries[entries.length - 1];
          const hasPartial = lastEntry ? !lastEntry.isFinal : false;
          setState((prev) => ({
            ...prev,
            entries: entries,
            isProcessing: hasPartial,
            hasPartialText: hasPartial,
          }));
        });

        process.on(
          "status",
          ({ status, message }: { status: string; message: string }) => {
            setState((prev) => ({
              ...prev,
              status: message || status,
            }));

            if (status === "recording") {
              setState((prev) => ({
                ...prev,
                isRecording: true,
                status: "🎤 Listening...",
              }));
            }
          },
        );

        process.on("error", (error: Error) => {
          console.error("Voice Input error:", error);
          setState((prev) => ({
            ...prev,
            isRecording: false,
            error: error.message,
            status: "Error",
          }));
          showToast({
            style: Toast.Style.Failure,
            title: "Error",
            message: error.message,
          });
        });

        process.on("close", (code: number | null) => {
          console.log("Voice Input: Process closed with code:", code);
          setState((prev) => ({
            ...prev,
            isRecording: false,
            status: "Stopped",
            audioLevel: 0,
          }));
        });

        process.on("level", (level: number) => {
          setState((prev) => ({
            ...prev,
            audioLevel: level,
          }));
        });

        process.on("spectrum", (spectrum: number[]) => {
          setState((prev) => ({
            ...prev,
            spectrum: spectrum,
          }));
        });

        // Use voice-single command for better accuracy (same as whisper-realtime voice)
        // Two-Pass処理: Partial(100ms/5秒窓/beam=1) + Final(beam=5)
        const startOptions: StartOptions = {
          useVoiceSingle: true, // Use voice-single for AquaVoice-style input
          voiceSingleStep: 1000, // 基本ステップ（Two-Pass時は上書きされる）
          voiceSingleLength: 20000, // 20秒の窓
          voiceSingleKeep: 20000, // コンテキスト保持（文の途切れ防止）
          voiceSingleMaxTokens: 256, // 約256文字まで対応（精度向上）
          // Two-Pass処理（Zoom/Google Meet方式）
          twoPass: true,
          partialStep: 10,
          partialWindow: 20,
          partialBeam: 1, // greedy search（高速）
          finalBeam: 8, // beam search（高精度）
          // 音声前処理
          noiseReduction: false, // DeepFilterNetノイズ除去（FFmpeg必要、デフォルト無効）
          vadThreshold: parseFloat(preferences.vadThreshold || "0.5"), // VAD閾値（設定から取得）
          // Utterance検出（発話確定）
          utteranceSilence: parseFloat(preferences.utteranceSilence || "0.8"), // 無音判定時間（設定から取得）
        };

        process.start(startOptions);

        // Check if process actually started
        const isRunning = process.getIsRunning();
        await showToast({
          style: isRunning ? Toast.Style.Success : Toast.Style.Failure,
          title: isRunning ? "🎤 Recording" : "Failed to start",
          message: isRunning
            ? "Speak now... (Enter to finish)"
            : "Process did not start",
        });

        setState({
          entries: [],
          isRecording: true,
          status: "🎤 Listening...",
          error: null,
          audioLevel: 0,
          spectrum: [0, 0, 0, 0, 0, 0, 0, 0],
          isProcessing: false,
          hasPartialText: false,
        });
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        setState((prev) => ({
          ...prev,
          error: message,
          status: "Error",
        }));
        await showToast({
          style: Toast.Style.Failure,
          title: "Error in startRecording",
          message: message,
        });
    }
  }, [preferences.whisperRealtimePath]);

  // Auto-start recording when component mounts
  useEffect(() => {
    if (hasStarted.current) return;
    hasStarted.current = true;

    startRecording();

    // Cleanup on unmount
    return () => {
      const process = getWhisperProcess();
      if (process.getIsRunning()) {
        process.stop();
      }
      // Also kill any zombie processes
      killAllWhisperProcesses();
    };
  }, [startRecording]);

  const typeAndClose = useCallback(async () => {
    const process = getWhisperProcess();

    // Get text including partial (since FINAL might not arrive yet)
    let finalText = process.getFullText(true);

    // Stop the process
    process.stop();

    // Wait a bit for FINAL to arrive
    await new Promise((resolve) => setTimeout(resolve, 300));

    // Try to get final text again
    const textAfterStop = process.getFullText(true);
    if (textAfterStop) {
      finalText = textAfterStop;
    }

    // Apply dictionary (including hallucination filter)
    try {
      const dictionary = await loadDictionary();
      finalText = applyDictionary(finalText, dictionary);
    } catch (error) {
      console.error("Failed to apply dictionary:", error);
    }

    if (!finalText.trim()) {
      await showToast({
        style: Toast.Style.Failure,
        title: "Nothing to type",
        message: "No speech detected",
      });
      await popToRoot();
      return;
    }

    // Show what we're about to paste
    await showToast({
      style: Toast.Style.Success,
      title: "Pasting text",
      message: finalText.substring(0, 30) + "...",
    });

    // Close Raycast window first to return focus to the previous app
    await closeMainWindow();

    // Small delay to ensure window is closed and focus is returned
    await new Promise((resolve) => setTimeout(resolve, 200));

    // Use paste instead of AppleScript for reliability
    await Clipboard.paste(finalText);

    // Clean up and reset state
    resetWhisperProcess();
    await killAllWhisperProcesses();

    // Reset hasStarted so next time voice input opens, it will auto-start
    hasStarted.current = false;

    // Pop to root to fully reset the component for next use
    await popToRoot();
  }, []);

  const pasteAndClose = useCallback(async () => {
    const process = getWhisperProcess();
    process.stop();

    // Get final text (excluding partial)
    let finalText = process.getFullText(false);

    // Apply dictionary (including hallucination filter)
    try {
      const dictionary = await loadDictionary();
      finalText = applyDictionary(finalText, dictionary);
    } catch (error) {
      console.error("Failed to apply dictionary:", error);
    }

    if (!finalText.trim()) {
      await showToast({
        style: Toast.Style.Failure,
        title: "Nothing to paste",
        message: "No speech detected",
      });
      await popToRoot();
      return;
    }

    // Close Raycast window first to return focus to the previous app
    await closeMainWindow();

    // Small delay to ensure window is closed and focus is returned
    await new Promise((resolve) => setTimeout(resolve, 100));

    // Paste the text to the frontmost application
    await Clipboard.paste(finalText);

    // Clean up
    resetWhisperProcess();
    await killAllWhisperProcesses();
  }, []);

  const cancelAndClose = useCallback(async () => {
    const process = getWhisperProcess();
    process.stop();
    resetWhisperProcess();
    await killAllWhisperProcesses();
    await popToRoot();
  }, []);

  const copyToClipboard = useCallback(async () => {
    const process = getWhisperProcess();
    process.stop();

    let finalText = process.getFullText(false);

    // Apply dictionary (including hallucination filter)
    try {
      const dictionary = await loadDictionary();
      finalText = applyDictionary(finalText, dictionary);
    } catch (error) {
      console.error("Failed to apply dictionary:", error);
    }

    if (!finalText.trim()) {
      await showToast({
        style: Toast.Style.Failure,
        title: "Nothing to copy",
        message: "No speech detected",
      });
      resetWhisperProcess();
      await killAllWhisperProcesses();
      await popToRoot();
      return;
    }

    await Clipboard.copy(finalText);
    await showToast({
      style: Toast.Style.Success,
      title: "Copied to clipboard",
    });

    resetWhisperProcess();
    await killAllWhisperProcesses();
    await popToRoot();
  }, []);

  const resetAndRestart = useCallback(async () => {
    // Stop current process
    const process = getWhisperProcess();
    process.stop();

    // Kill all zombie processes
    resetWhisperProcess();
    await killAllWhisperProcesses();

    // Reset state
    setState({
      entries: [],
      isRecording: false,
      status: "Resetting...",
      error: null,
      audioLevel: 0,
      spectrum: [0, 0, 0, 0, 0, 0, 0, 0],
      isProcessing: false,
      hasPartialText: false,
    });

    await showToast({
      style: Toast.Style.Animated,
      title: "Resetting...",
      message: "Restarting recording",
    });

    // Wait a bit for cleanup to complete
    await new Promise((resolve) => setTimeout(resolve, 300));

    // Directly restart recording
    await startRecording();
  }, [startRecording]);

  // Generate audio level bar for visualization
  const generateLevelBar = (level: number) => {
    const barLength = 20;
    const filledLength = Math.round(level * barLength);
    const filled = "█".repeat(filledLength);
    const empty = "░".repeat(barLength - filledLength);
    return `\`${filled}${empty}\``;
  };

  // Generate formatted text with final parts in bold
  const generateFormattedText = () => {
    if (state.entries.length === 0) return "";

    return state.entries
      .map((entry) => {
        if (!entry.text) return "";
        // 確定部分は太字、partial部分は通常テキスト
        return entry.isFinal ? `**${entry.text}**` : entry.text;
      })
      .filter(Boolean)
      .join("");
  };

  // Generate compact markdown for voice input
  const generateMarkdown = () => {
    const lines: string[] = [];

    lines.push("# 🎙️ Voice Input");
    lines.push("");

    if (state.error) {
      lines.push(`**Error:** ${state.error}`);
    } else if (state.isRecording) {
      lines.push(`**${state.status}**`);
      lines.push("");

      // Audio level indicator
      lines.push("**Audio Level:**");
      lines.push(generateLevelBar(state.audioLevel));
      lines.push("");

      const formattedText = generateFormattedText();
      if (formattedText) {
        lines.push("---");
        lines.push("");
        if (state.isProcessing) {
          // 処理中：テキストの後に処理中インジケーターを表示
          lines.push(`> ${formattedText}`);
          lines.push("");
          lines.push("*🔄 文字起こし中...*");
        } else {
          // 完了：確定したテキストを表示
          lines.push(`> ${formattedText}`);
          lines.push("");
          lines.push("*✅ 文字起こし完了*");
        }
      } else {
        lines.push("*Speak now...*");
      }

      lines.push("");
      lines.push("---");
      lines.push("");
      lines.push(
        "**⏎** Type • **⌘V** Paste • **⌘C** Copy • **⌘R** Reset • **Esc** Cancel",
      );
    } else {
      lines.push(`**Status:** ${state.status}`);
      const formattedText = generateFormattedText();
      if (formattedText) {
        lines.push("");
        lines.push(`> ${formattedText}`);
      }
    }

    return lines.join("\n");
  };

  return (
    <Detail
      markdown={generateMarkdown()}
      isLoading={state.status === "Starting..."}
      actions={
        <ActionPanel>
          <ActionPanel.Section title="Input">
            <Action
              title="Type Text"
              icon={Icon.Text}
              onAction={typeAndClose}
            />
            <Action
              title="Paste Text"
              icon={Icon.Clipboard}
              onAction={pasteAndClose}
              shortcut={{ modifiers: ["cmd"], key: "v" }}
            />
          </ActionPanel.Section>
          <ActionPanel.Section title="Other">
            <Action
              title="Copy to Clipboard"
              icon={Icon.CopyClipboard}
              onAction={copyToClipboard}
              shortcut={{ modifiers: ["cmd"], key: "c" }}
            />
            <Action
              title="Reset"
              icon={Icon.ArrowClockwise}
              onAction={resetAndRestart}
              shortcut={{ modifiers: ["cmd"], key: "r" }}
            />
            <Action
              title="Cancel"
              icon={Icon.XMarkCircle}
              onAction={cancelAndClose}
              shortcut={{ modifiers: [], key: "escape" }}
            />
          </ActionPanel.Section>
        </ActionPanel>
      }
    />
  );
}
