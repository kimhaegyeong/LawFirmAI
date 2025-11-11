/**
 * 타이핑 효과 훅
 * 버퍼에 저장된 텍스트를 한 글자씩 표시하는 효과를 제공합니다.
 */
import { useState, useEffect, useRef } from 'react';
import logger from '../utils/logger';

interface UseTypingEffectOptions {
  speed?: number; // 타이핑 속도 (ms)
  enabled?: boolean; // 타이핑 효과 활성화 여부
}

interface UseTypingEffectResult {
  displayed: string;
  isComplete: boolean; // 타이핑 효과가 완료되었는지 여부
}

export function useTypingEffect(
  buffer: string,
  options: UseTypingEffectOptions = {}
): UseTypingEffectResult {
  const { speed = 30, enabled = true } = options;
  const [displayed, setDisplayed] = useState('');
  const [isComplete, setIsComplete] = useState(false);
  const frameRef = useRef<number | null>(null);
  const lastTimeRef = useRef<number>(0);
  const displayedLengthRef = useRef<number>(0);
  const bufferRef = useRef<string>(buffer);
  const animationActiveRef = useRef<boolean>(false);
  const previousBufferLengthRef = useRef<number>(0);

  useEffect(() => {
    bufferRef.current = buffer;
  }, [buffer]);

  useEffect(() => {
    if (import.meta.env.DEV) {
      logger.debug('[TypingEffect] Buffer updated:', buffer.length, 'Displayed:', displayedLengthRef.current, 'Enabled:', enabled);
    }

    if (!enabled) {
      setDisplayed(buffer);
      displayedLengthRef.current = buffer.length;
      previousBufferLengthRef.current = buffer.length;
      setIsComplete(true);
      if (frameRef.current) {
        cancelAnimationFrame(frameRef.current);
        frameRef.current = null;
      }
      animationActiveRef.current = false;
      return;
    }

    if (buffer.length === 0) {
      setDisplayed('');
      displayedLengthRef.current = 0;
      previousBufferLengthRef.current = 0;
      setIsComplete(true);
      if (frameRef.current) {
        cancelAnimationFrame(frameRef.current);
        frameRef.current = null;
      }
      animationActiveRef.current = false;
      return;
    }

    // 버퍼 증가량 추적 (타이핑 효과는 한 글자씩 유지)
    previousBufferLengthRef.current = buffer.length;

    setDisplayed(prev => {
      if (prev.length !== displayedLengthRef.current) {
        const synced = buffer.slice(0, displayedLengthRef.current);
        if (import.meta.env.DEV) {
          logger.debug('[TypingEffect] Syncing displayed:', prev.length, '->', displayedLengthRef.current);
        }
        return synced;
      }
      return prev;
    });

    const animate = (currentTime: number) => {
      const currentBuffer = bufferRef.current;
      const currentDisplayedLength = displayedLengthRef.current;

      if (currentDisplayedLength < currentBuffer.length) {
        const elapsed = currentTime - lastTimeRef.current;
        if (elapsed >= speed) {
          // 항상 한 글자씩만 표시 (자연스러운 타이핑 효과)
          const nextLength = currentDisplayedLength + 1;
          const next = currentBuffer.slice(0, nextLength);
          
          setDisplayed(next);
          displayedLengthRef.current = nextLength;
          lastTimeRef.current = currentTime;
          setIsComplete(false);
        }

        frameRef.current = requestAnimationFrame(animate);
      } else {
        animationActiveRef.current = false;
        frameRef.current = null;
        setIsComplete(true);
      }
    };

    // 버퍼가 업데이트되어도 타이핑 효과가 지속되도록 보장
    const bufferIncreaseAmount = buffer.length - previousBufferLengthRef.current;
    if (bufferIncreaseAmount > 0 && displayedLengthRef.current < buffer.length) {
      // 타이핑 효과가 진행 중이 아니면 시작
      if (!animationActiveRef.current && displayedLengthRef.current < buffer.length) {
        animationActiveRef.current = true;
        lastTimeRef.current = performance.now();
        setIsComplete(false);
        if (!frameRef.current) {
          frameRef.current = requestAnimationFrame(animate);
        }
      }
    }

    if (displayedLengthRef.current < buffer.length && !animationActiveRef.current) {
      animationActiveRef.current = true;
      lastTimeRef.current = performance.now();
      setIsComplete(false);
      if (!frameRef.current) {
        frameRef.current = requestAnimationFrame(animate);
      }
    }

    if (!frameRef.current && displayedLengthRef.current < buffer.length) {
      animationActiveRef.current = true;
      lastTimeRef.current = performance.now();
      setIsComplete(false);
      frameRef.current = requestAnimationFrame(animate);
    }

    if (displayedLengthRef.current >= buffer.length && buffer.length > 0) {
      setIsComplete(true);
    }

    return () => {
      if (frameRef.current) {
        cancelAnimationFrame(frameRef.current);
        frameRef.current = null;
      }
      animationActiveRef.current = false;
    };
  }, [buffer, speed, enabled]);

  return { displayed, isComplete };
}

