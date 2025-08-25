import React, { useState, useEffect } from 'react';

interface TypedTextProps {
  text: string;
  speed?: number;
  delay?: number;
  className?: string;
  onComplete?: () => void;
  children?: React.ReactNode;
  hideCursor?: boolean;  // New prop to hide cursor
  verticalMode?: boolean; // New prop for vertical typing
}

export const TypedText: React.FC<TypedTextProps> = ({
  text,
  speed = 30, // milliseconds per character
  delay = 0, // delay before starting
  className = '',
  onComplete,
  children,
  hideCursor = false,
  verticalMode = true // Enable vertical by default
}) => {
  const [displayText, setDisplayText] = useState('');
  const [isComplete, setIsComplete] = useState(false);
  const [showCursor, setShowCursor] = useState(!hideCursor);

  useEffect(() => {
    if (!text) {
      setIsComplete(true);
      return;
    }

    let timeoutId: NodeJS.Timeout;
    let currentIndex = 0;

    const startTyping = () => {
      const typeCharacter = () => {
        if (currentIndex < text.length) {
          const currentChar = text[currentIndex];
          setDisplayText(text.slice(0, currentIndex + 1));
          currentIndex++;
          
          // In vertical mode, speed up line breaks for natural flow
          let nextDelay = speed;
          if (verticalMode && (currentChar === '\n' || currentChar === '\r')) {
            nextDelay = speed * 0.3; // Faster line breaks
          }
          
          timeoutId = setTimeout(typeCharacter, nextDelay);
        } else {
          setIsComplete(true);
          if (hideCursor) {
            setShowCursor(false);
          }
          onComplete?.();
        }
      };
      typeCharacter();
    };

    // Start typing after delay
    timeoutId = setTimeout(startTyping, delay);

    return () => {
      clearTimeout(timeoutId);
    };
  }, [text, speed, delay, onComplete, hideCursor, verticalMode]);

  // Cursor blinking effect - only if not hidden and not complete
  useEffect(() => {
    if (hideCursor || isComplete) return;

    const cursorInterval = setInterval(() => {
      setShowCursor(prev => !prev);
    }, 500);

    return () => clearInterval(cursorInterval);
  }, [hideCursor, isComplete]);

  return (
    <span className={`${className} ${verticalMode ? 'whitespace-pre-wrap' : ''}`}>
      {displayText}
      {!isComplete && !hideCursor && showCursor && (
        <span className="inline-block w-0.5 h-5 bg-current ml-0.5 animate-pulse" />
      )}
      {isComplete && children}
    </span>
  );
};

interface TypedParagraphProps {
  text: string;
  speed?: number;
  delay?: number;
  className?: string;
  onComplete?: () => void;
}

export const TypedParagraph: React.FC<TypedParagraphProps> = ({
  text,
  speed = 20,
  delay = 0,
  className = '',
  onComplete
}) => {
  return (
    <div className={className}>
      <TypedText 
        text={text} 
        speed={speed} 
        delay={delay} 
        onComplete={onComplete}
      />
    </div>
  );
};

// For faster typing in code blocks or technical content
export const TypedCode: React.FC<TypedTextProps> = (props) => {
  return (
    <TypedText 
      {...props} 
      speed={props.speed || 15} // Faster for code content
      className={`font-mono ${props.className || ''}`}
    />
  );
};

// Batch typing for lists
interface TypedListProps {
  items: string[];
  itemSpeed?: number;
  itemDelay?: number;
  className?: string;
  onComplete?: () => void;
}

export const TypedList: React.FC<TypedListProps> = ({
  items,
  itemSpeed = 25,
  itemDelay = 200,
  className = '',
  onComplete
}) => {
  const [currentItem, setCurrentItem] = useState(0);
  const [completedItems, setCompletedItems] = useState(0);

  const handleItemComplete = () => {
    setCompletedItems(prev => {
      const newCount = prev + 1;
      if (newCount === items.length) {
        onComplete?.();
      } else {
        setTimeout(() => setCurrentItem(newCount), itemDelay);
      }
      return newCount;
    });
  };

  return (
    <div className={className}>
      {items.map((item, index) => (
        <div key={index} className="mb-2">
          {index <= currentItem && (
            <TypedText
              text={item}
              speed={itemSpeed}
              onComplete={index === currentItem ? handleItemComplete : undefined}
            />
          )}
        </div>
      ))}
    </div>
  );
};