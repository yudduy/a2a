import { useState, useId } from 'react';
import { Button } from '@/components/ui/button';
import {
  SquarePen,
  Send,
  StopCircle,
  Search,
} from 'lucide-react';
import { Textarea } from '@/components/ui/textarea';

interface InputFormProps {
  onSubmit: (inputValue: string) => void;
  onCancel: () => void;
  isLoading: boolean;
  hasHistory: boolean;
  placeholder?: string;
  onReset?: () => void;
}

export const InputForm: React.FC<InputFormProps> = ({
  onSubmit,
  onCancel,
  isLoading,
  hasHistory,
  placeholder,
  onReset,
}) => {
  const [internalInputValue, setInternalInputValue] = useState('');
  const [isComposing, setIsComposing] = useState(false);
  const textareaId = useId();
  const helperTextId = useId();

  const handleInternalSubmit = (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    if (!internalInputValue.trim()) return;
    onSubmit(internalInputValue);
    setInternalInputValue('');
  };

  const handleInternalKeyDown = (
    e: React.KeyboardEvent<HTMLTextAreaElement>
  ) => {
    // Check for IME composition using both the native event flag and our state
    if ((e.nativeEvent as any).isComposing || isComposing) {
      return;
    }
    
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleInternalSubmit();
    }
  };

  const handleCompositionStart = () => {
    setIsComposing(true);
  };

  const handleCompositionEnd = () => {
    setIsComposing(false);
  };

  const isSubmitDisabled = !internalInputValue.trim() || isLoading;

  return (
    <form
      onSubmit={handleInternalSubmit}
      className="flex flex-col gap-3 p-4"
    >
      {/* Research Input Area */}
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-3 text-sm text-neutral-400">
          <Search className="h-4 w-4" aria-hidden="true" />
          <span id={helperTextId}>Enter your research query to begin intelligent analysis</span>
        </div>
      </div>

      <div
        className={`flex flex-row items-center justify-between text-white rounded-2xl ${
          hasHistory ? 'rounded-br-sm' : ''
        } break-words min-h-7 bg-neutral-700 px-6 py-4 border border-neutral-600`}
      >
        <Textarea
          id={textareaId}
          value={internalInputValue}
          onChange={(e) => setInternalInputValue(e.target.value)}
          onKeyDown={handleInternalKeyDown}
          onCompositionStart={handleCompositionStart}
          onCompositionEnd={handleCompositionEnd}
          placeholder={placeholder || "What would you like to research? e.g., 'Latest developments in quantum computing and their commercial applications'"}
          className="w-full text-neutral-100 placeholder-neutral-500 resize-none border-0 focus:outline-none focus:ring-0 outline-none focus-visible:ring-0 shadow-none text-base min-h-[80px] max-h-[200px]"
          rows={3}
          aria-label="Research query input"
          aria-describedby={helperTextId}
        />
        <div className="ml-4 flex-shrink-0">
          {isLoading ? (
            <Button
              type="button"
              variant="ghost"
              size="icon"
              className="text-red-500 hover:text-red-400 hover:bg-red-500/10 p-3 cursor-pointer rounded-full transition-all duration-200"
              onClick={onCancel}
              aria-label="Stop research"
              title="Stop the current research process"
            >
              <StopCircle className="h-6 w-6" aria-hidden="true" />
            </Button>
          ) : (
            <Button
              type="submit"
              variant="ghost"
              className={`${
                isSubmitDisabled
                  ? 'text-neutral-500'
                  : 'text-blue-500 hover:text-blue-400 hover:bg-blue-500/10'
              } p-3 cursor-pointer rounded-full transition-all duration-200`}
              disabled={isSubmitDisabled}
              aria-label="Submit research query"
              title="Submit your research query to begin analysis"
            >
              <Send className="h-6 w-6" aria-hidden="true" />
            </Button>
          )}
        </div>
      </div>

      {hasHistory && (
        <div className="flex justify-end">
          <Button
            className="bg-neutral-700 hover:bg-neutral-600 border-neutral-600 text-neutral-300 cursor-pointer rounded-xl px-4 py-2"
            variant="default"
            onClick={onReset || (() => window.location.reload())}
            aria-label="Start new research"
            title="Clear current research and start a new one"
          >
            <SquarePen size={16} className="mr-2" aria-hidden="true" />
            New Research
          </Button>
        </div>
      )}
    </form>
  );
};
