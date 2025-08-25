import React, { useState, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';

interface TypedMarkdownProps {
  children: string;
  components?: any;
  speed?: number;
  delay?: number;
  className?: string;
  hideCursor?: boolean;
  verticalMode?: boolean;
  onTypingComplete?: () => void;
  enableTyping?: boolean;
}

export const TypedMarkdown: React.FC<TypedMarkdownProps> = ({
  children,
  components,
  speed = 25,
  delay = 0,
  className = '',
  hideCursor = true, // Hide cursor by default for cleaner markdown
  verticalMode = true,
  onTypingComplete,
  enableTyping = true,
}) => {
  const [isTyping, setIsTyping] = useState(enableTyping);
  const [displayText, setDisplayText] = useState(enableTyping ? '' : children);

  useEffect(() => {
    if (!children || !enableTyping) {
      setIsTyping(false);
      setDisplayText(children);
      // Call completion callback immediately if typing is disabled
      if (onTypingComplete && !enableTyping) {
        setTimeout(onTypingComplete, 0);
      }
      return;
    }

    let timeoutId: NodeJS.Timeout;
    let currentIndex = 0;

    const startTyping = () => {
      const typeCharacter = () => {
        if (currentIndex < children.length) {
          const currentChar = children[currentIndex];
          setDisplayText(children.slice(0, currentIndex + 1));
          currentIndex++;
          
          // Speed up line breaks for natural flow
          let nextDelay = speed;
          if (verticalMode && (currentChar === '\n' || currentChar === '\r')) {
            nextDelay = speed * 0.3;
          }
          
          timeoutId = setTimeout(typeCharacter, nextDelay);
        } else {
          setIsTyping(false);
          // Call completion callback when typing is finished
          if (onTypingComplete) {
            onTypingComplete();
          }
        }
      };
      typeCharacter();
    };

    // Start typing after delay
    timeoutId = setTimeout(startTyping, delay);

    return () => {
      clearTimeout(timeoutId);
    };
  }, [children, speed, delay, verticalMode, onTypingComplete, enableTyping]);

  if (!children) {
    return null;
  }

  return (
    <div className={className}>
      {isTyping ? (
        <div className={verticalMode ? 'whitespace-pre-wrap' : ''}>
          <ReactMarkdown components={components}>
            {displayText}
          </ReactMarkdown>
          {!hideCursor && (
            <span className="inline-block w-0.5 h-5 bg-current ml-1 animate-pulse" />
          )}
        </div>
      ) : (
        <ReactMarkdown components={components}>
          {children}
        </ReactMarkdown>
      )}
    </div>
  );
};

// For faster typing in code/technical content
export const TypedCodeMarkdown: React.FC<TypedMarkdownProps> = (props) => {
  return (
    <TypedMarkdown 
      {...props} 
      speed={props.speed || 15} // Faster for code content
      hideCursor={true}
      verticalMode={true}
    />
  );
};

// For thinking sections with proper styling
export const TypedThinkingMarkdown: React.FC<TypedMarkdownProps> = (props) => {
  return (
    <TypedMarkdown 
      {...props} 
      speed={props.speed || 20}
      hideCursor={true}
      verticalMode={true}
      className={`font-mono text-sm leading-relaxed ${props.className || ''}`}
    />
  );
};