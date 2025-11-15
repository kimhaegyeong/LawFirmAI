/**
 * 사이드바 상태 관리 훅
 */
import { useState, useCallback } from 'react';
import type { ChatMessage } from '../types/chat';

type ReferenceType = 'all' | 'law' | 'precedent' | 'decision' | 'interpretation' | 'regulation';

export function useSidebar() {
  const [documentSidebarOpen, setDocumentSidebarOpen] = useState(false);
  const [selectedDocumentIndex, setSelectedDocumentIndex] = useState<number | null>(null);
  const [selectedMessageForDocument, setSelectedMessageForDocument] = useState<ChatMessage | null>(null);
  
  const [referencesSidebarOpen, setReferencesSidebarOpen] = useState(false);
  const [selectedMessageForReferences, setSelectedMessageForReferences] = useState<ChatMessage | null>(null);
  const [selectedReferenceType, setSelectedReferenceType] = useState<ReferenceType>('all');

  const openDocumentSidebar = useCallback((message: ChatMessage, index: number) => {
    setSelectedMessageForDocument(message);
    setSelectedDocumentIndex(index);
    setDocumentSidebarOpen(true);
  }, []);

  const closeDocumentSidebar = useCallback(() => {
    setDocumentSidebarOpen(false);
    setSelectedDocumentIndex(null);
    setSelectedMessageForDocument(null);
  }, []);

  const openReferencesSidebar = useCallback((message: ChatMessage, selectedType: ReferenceType = 'all') => {
    setSelectedMessageForReferences(message);
    setSelectedReferenceType(selectedType);
    setReferencesSidebarOpen(true);
  }, []);

  const closeReferencesSidebar = useCallback(() => {
    setReferencesSidebarOpen(false);
    setSelectedMessageForReferences(null);
    setSelectedReferenceType('all');
  }, []);

  const updateSelectedMessageForDocument = useCallback((updater: (message: ChatMessage | null) => ChatMessage | null) => {
    setSelectedMessageForDocument(updater);
  }, []);

  return {
    // Document Sidebar
    documentSidebarOpen,
    selectedDocumentIndex,
    selectedMessageForDocument,
    openDocumentSidebar,
    closeDocumentSidebar,
    updateSelectedMessageForDocument,
    
    // References Sidebar
    referencesSidebarOpen,
    selectedMessageForReferences,
    selectedReferenceType,
    openReferencesSidebar,
    closeReferencesSidebar,
  };
}

