import React from 'react';
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from './collapsible';
import { Badge } from './badge';
import { ChevronRight, Brain } from 'lucide-react';
import { cn } from '@/lib/utils';
import { TypedThinkingMarkdown } from './typed-markdown';
import { ThinkingSection } from '@/types/messages';

interface CollapsibleThinkingProps {
  section: ThinkingSection;
  isExpanded: boolean;
  onToggle: () => void;
  hasTypingAnimation?: boolean;
  typingSpeed?: number;
  className?: string;
  onTypingComplete?: () => void;
}

export const CollapsibleThinking: React.FC<CollapsibleThinkingProps> = ({
  section,
  isExpanded,
  onToggle,
  hasTypingAnimation = false,
  typingSpeed = 20,
  className = '',
  onTypingComplete,
}) => {
  return (
    <div className={cn(
      'border border-blue-500/30 bg-blue-900/10 rounded-lg overflow-hidden my-3',
      className
    )}>
      <Collapsible open={isExpanded} onOpenChange={onToggle}>
        <CollapsibleTrigger asChild>
          <button className="w-full px-4 py-3 hover:bg-blue-900/20 transition-all duration-200 text-left focus:outline-none focus:ring-2 focus:ring-blue-500 focus:bg-blue-900/20">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className="p-1.5 rounded-md bg-blue-700/50 flex-shrink-0">
                  <Brain className="h-3.5 w-3.5 text-blue-300" />
                </div>
                <div className="flex flex-col gap-1">
                  <span className="font-medium text-blue-200 text-sm">
                    thinking...
                  </span>
                </div>
                <Badge 
                  variant="secondary" 
                  className="bg-blue-800/30 text-blue-200 border-blue-500/20 text-xs font-medium"
                >
                  {section.charLength} chars
                </Badge>
              </div>
              <ChevronRight
                className={cn(
                  'h-4 w-4 text-blue-400 transition-transform duration-200 flex-shrink-0',
                  isExpanded && 'rotate-90'
                )}
              />
            </div>
          </button>
        </CollapsibleTrigger>
        <CollapsibleContent
          className={cn(
            'data-[state=open]:animate-collapsible-down data-[state=closed]:animate-collapsible-up'
          )}
        >
          <div className="px-4 pb-4 border-t border-blue-500/20 bg-blue-950/20">
            <div className="pt-4">
              {hasTypingAnimation ? (
                <TypedThinkingMarkdown 
                  speed={typingSpeed}
                  className="text-blue-100"
                  onTypingComplete={onTypingComplete}
                >
                  {section.content}
                </TypedThinkingMarkdown>
              ) : (
                <div className="font-mono text-sm text-blue-100 leading-relaxed whitespace-pre-wrap">
                  {section.content}
                </div>
              )}
            </div>
          </div>
        </CollapsibleContent>
      </Collapsible>
    </div>
  );
};

// Helper component for multiple thinking sections
interface ThinkingSectionsProps {
  sections: ThinkingSection[];
  expandedSections: Set<string>;
  onToggleSection: (sectionId: string) => void;
  hasTypingAnimation?: boolean;
  typingSpeed?: number;
  className?: string;
  onTypingComplete?: () => void;
}

export const ThinkingSections: React.FC<ThinkingSectionsProps> = ({
  sections,
  expandedSections,
  onToggleSection,
  hasTypingAnimation = false,
  typingSpeed = 20,
  className = '',
  onTypingComplete,
}) => {
  if (sections.length === 0) {
    return null;
  }

  return (
    <div className={cn('space-y-2', className)}>
      {sections.map((section, index) => (
        <CollapsibleThinking
          key={section.id}
          section={section}
          isExpanded={expandedSections.has(section.id)}
          onToggle={() => onToggleSection(section.id)}
          hasTypingAnimation={hasTypingAnimation && index === 0} // Only first section gets typing animation
          typingSpeed={typingSpeed}
          onTypingComplete={index === 0 ? onTypingComplete : undefined} // Only trigger callback for first section
        />
      ))}
    </div>
  );
};

// Simple thinking indicator for collapsed state
export const ThinkingIndicator: React.FC<{ 
  count: number; 
  onClick?: () => void;
  className?: string;
}> = ({ count, onClick, className = '' }) => {
  if (count === 0) return null;

  return (
    <button
      onClick={onClick}
      className={cn(
        'inline-flex items-center gap-2 px-3 py-1.5 rounded-full',
        'bg-blue-900/20 border border-blue-500/30 text-blue-200',
        'hover:bg-blue-900/30 transition-all duration-200',
        'focus:outline-none focus:ring-2 focus:ring-blue-500',
        'text-sm font-medium',
        className
      )}
    >
      <Brain className="h-3 w-3" />
      <span>
        {count} thinking section{count !== 1 ? 's' : ''}
      </span>
    </button>
  );
};