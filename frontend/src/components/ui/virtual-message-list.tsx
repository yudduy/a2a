/**
 * VirtualMessageList - High-performance virtual scrolling for chat messages
 * 
 * This component provides:
 * - Virtual scrolling for thousands of messages
 * - Efficient memory usage
 * - Smooth scrolling performance
 * - Auto-scroll to bottom
 * - Message caching and recycling
 */

import React, { memo, useCallback, useEffect, useRef, useState, useMemo } from 'react';
import { FixedSizeList as List, areEqual } from 'react-window';
import { RoutedMessage } from '@/types/parallel';
import { cn } from '@/lib/utils';

interface VirtualMessageListProps {
  messages: RoutedMessage[];
  height: number;
  width?: number | string;
  itemHeight: number;
  overscan?: number;
  autoScrollToBottom?: boolean;
  onMessageClick?: (message: RoutedMessage) => void;
  className?: string;
  renderMessage: (message: RoutedMessage, index: number) => React.ReactNode;
}

interface MessageItemProps {
  index: number;
  style: React.CSSProperties;
  data: {
    messages: RoutedMessage[];
    renderMessage: (message: RoutedMessage, index: number) => React.ReactNode;
    onMessageClick?: (message: RoutedMessage) => void;
  };
}

// Memoized message item component
const MessageItem = memo<MessageItemProps>(({ index, style, data }) => {
  const { messages, renderMessage, onMessageClick } = data;
  const message = messages[index];

  if (!message) {
    return (
      <div style={style} className="flex items-center justify-center text-neutral-500">
        <span className="text-sm">Message not available</span>
      </div>
    );
  }

  const handleClick = useCallback(() => {
    onMessageClick?.(message);
  }, [message, onMessageClick]);

  return (
    <div 
      style={style} 
      className="px-1"
      onClick={handleClick}
    >
      {renderMessage(message, index)}
    </div>
  );
}, areEqual);

MessageItem.displayName = 'MessageItem';

const VirtualMessageList = memo<VirtualMessageListProps>(({
  messages,
  height,
  width = '100%',
  itemHeight,
  overscan = 5,
  autoScrollToBottom = true,
  onMessageClick,
  className,
  renderMessage,
}) => {
  const listRef = useRef<List>(null);
  const [isUserScrolling, setIsUserScrolling] = useState(false);
  const userScrollTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const lastMessageCountRef = useRef(messages.length);

  // Memoized item data to prevent unnecessary re-renders
  const itemData = useMemo(() => ({
    messages,
    renderMessage,
    onMessageClick,
  }), [messages, renderMessage, onMessageClick]);

  // Auto-scroll to bottom when new messages arrive (if user isn't manually scrolling)
  useEffect(() => {
    if (autoScrollToBottom && !isUserScrolling && messages.length > lastMessageCountRef.current) {
      listRef.current?.scrollToItem(messages.length - 1, 'end');
    }
    lastMessageCountRef.current = messages.length;
  }, [messages.length, autoScrollToBottom, isUserScrolling]);

  // Handle scroll events to detect user scrolling
  const handleScroll = useCallback(() => {
    setIsUserScrolling(true);
    
    // Clear existing timeout
    if (userScrollTimeoutRef.current) {
      clearTimeout(userScrollTimeoutRef.current);
    }
    
    // Reset user scrolling state after 2 seconds of no scrolling
    userScrollTimeoutRef.current = setTimeout(() => {
      setIsUserScrolling(false);
    }, 2000);
  }, []);

  // Cleanup timeout on unmount
  useEffect(() => {
    return () => {
      if (userScrollTimeoutRef.current) {
        clearTimeout(userScrollTimeoutRef.current);
      }
    };
  }, []);

  // Scroll to bottom function for external use
  const scrollToBottom = useCallback(() => {
    if (messages.length > 0) {
      listRef.current?.scrollToItem(messages.length - 1, 'end');
      setIsUserScrolling(false);
    }
  }, [messages.length]);

  // Expose scroll to bottom function
  const scrollAPI = {
    scrollToBottom,
    scrollToItem: (index: number, align?: 'auto' | 'smart' | 'center' | 'end' | 'start') => {
      listRef.current?.scrollToItem(index, align);
    },
  };
  
  React.useImperativeHandle(listRef, () => scrollAPI as any);

  return (
    <div className={cn("relative", className)}>
      {messages.length === 0 ? (
        <div 
          className="flex items-center justify-center text-neutral-500"
          style={{ height }}
        >
          <div className="text-center">
            <div className="text-sm font-medium mb-1">No messages yet</div>
            <div className="text-xs text-neutral-600">Messages will appear here as they arrive</div>
          </div>
        </div>
      ) : (
        <List
          ref={listRef}
          height={height}
          width={width}
          itemCount={messages.length}
          itemSize={itemHeight}
          itemData={itemData}
          overscanCount={overscan}
          onScroll={handleScroll}
          style={{
            scrollbarWidth: 'thin',
            scrollbarColor: 'rgb(115 115 115) transparent',
          }}
        >
          {MessageItem}
        </List>
      )}
      
      {/* Scroll to bottom button */}
      {isUserScrolling && messages.length > 0 && (
        <button
          onClick={scrollToBottom}
          className="absolute bottom-4 right-4 bg-blue-600 hover:bg-blue-700 text-white p-2 rounded-full shadow-lg transition-all duration-200 z-10"
          aria-label="Scroll to bottom"
        >
          <svg
            className="h-4 w-4"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 14l-7 7m0 0l-7-7m7 7V3" />
          </svg>
        </button>
      )}
    </div>
  );
});

VirtualMessageList.displayName = 'VirtualMessageList';

export { VirtualMessageList };
export default VirtualMessageList;