/**
 * 채팅 입력 컴포넌트
 */
import { Send, Paperclip } from 'lucide-react';
import { useState, useRef, useEffect, KeyboardEvent } from 'react';
import { FileAttachment } from '../common/FileAttachment';
import { fileToFileInfo, formatFileSize } from '../../utils/fileUtils';
import type { FileAttachment as FileAttachmentType } from '../../types/chat';

interface ChatInputProps {
  onSend: (message: string, attachments?: FileAttachmentType[]) => void;
  disabled?: boolean;
  isLoading?: boolean;
  resetTrigger?: number;
}

export function ChatInput({ onSend, disabled = false, isLoading = false, resetTrigger }: ChatInputProps) {
  const [message, setMessage] = useState('');
  const [attachments, setAttachments] = useState<FileAttachmentType[]>([]);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // resetTrigger가 변경되면 입력창 초기화
  useEffect(() => {
    if (resetTrigger !== undefined) {
      setMessage('');
      setAttachments([]);
      
      // 텍스트 영역 높이 초기화
      if (textareaRef.current) {
        textareaRef.current.style.height = 'auto';
      }
      
      // 파일 입력 초기화
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    }
  }, [resetTrigger]);

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files || []);
    const newAttachments = files.map(fileToFileInfo);
    setAttachments([...attachments, ...newAttachments]);
    
    // 파일 입력 초기화
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const removeAttachment = (id: string) => {
    setAttachments(attachments.filter((att) => att.id !== id));
  };

  const handleSend = () => {
    if ((message.trim() || attachments.length > 0) && !disabled && !isLoading) {
      onSend(message, attachments);
      setMessage('');
      setAttachments([]);
      
      // 텍스트 영역 높이 초기화
      if (textareaRef.current) {
        textareaRef.current.style.height = 'auto';
      }
    }
  };

  const handleKeyPress = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleTextareaChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setMessage(e.target.value);
    
    // 자동 높이 조절
    e.target.style.height = 'auto';
    e.target.style.height = `${Math.min(e.target.scrollHeight, 128)}px`;
  };

  return (
    <div className="bg-white border-t border-slate-200 px-6 py-4 flex-shrink-0">
      <div className="max-w-4xl mx-auto">
        {/* 파일 미리보기 */}
        {attachments.length > 0 && (
          <div className="mb-3 flex flex-wrap gap-2">
            {attachments.map((att) => (
              <FileAttachment
                key={att.id}
                attachment={att}
                onRemove={removeAttachment}
              />
            ))}
          </div>
        )}

        {/* 입력 박스 */}
        <div className="relative flex items-center gap-2 bg-white border-2 border-slate-300 rounded-2xl hover:border-slate-400 focus-within:border-blue-500 focus-within:shadow-lg transition-all">
          <input
            ref={fileInputRef}
            type="file"
            multiple
            onChange={handleFileUpload}
            className="hidden"
            accept="image/*,.pdf,.doc,.docx,.txt"
          />

          <button
            onClick={() => fileInputRef.current?.click()}
            className="ml-3 p-2.5 hover:bg-slate-100 rounded-lg transition-colors flex-shrink-0"
            title="파일 첨부"
            disabled={disabled || isLoading}
          >
            <Paperclip className="w-5 h-5 text-slate-600" />
          </button>

          <textarea
            ref={textareaRef}
            value={message}
            onChange={handleTextareaChange}
            onKeyPress={handleKeyPress}
            placeholder="메시지를 입력하세요..."
            className="flex-1 px-2 py-3.5 bg-transparent resize-none focus:outline-none text-slate-800 placeholder-slate-500 max-h-32"
            rows={1}
            style={{ minHeight: '44px' }}
            disabled={disabled || isLoading}
          />

          <button
            onClick={handleSend}
            disabled={(!message.trim() && attachments.length === 0) || disabled || isLoading}
            className="mr-3 p-2.5 bg-blue-600 hover:bg-blue-700 disabled:bg-slate-300 disabled:cursor-not-allowed rounded-lg transition-colors flex-shrink-0 shadow-sm"
            title="전송"
          >
            <Send className="w-5 h-5 text-white" />
          </button>
        </div>

        <div className="mt-2.5 text-xs text-slate-500 text-center">
          Shift + Enter로 줄바꿈, Enter로 전송
        </div>
      </div>
    </div>
  );
}

