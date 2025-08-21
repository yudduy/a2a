import { useState } from 'react';
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

  const handleInternalSubmit = (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    if (!internalInputValue.trim()) return;
    onSubmit(internalInputValue);
    setInternalInputValue('');
  };

  const handleInternalKeyDown = (
    e: React.KeyboardEvent<HTMLTextAreaElement>
  ) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleInternalSubmit();
    }
  };

  const isSubmitDisabled = !internalInputValue.trim() || isLoading;

  return (
    <form
      onSubmit={handleInternalSubmit}
      className="flex flex-col gap-3 p-4"
    >
      {/* Research Input Area */}
      <div className="flex items-center gap-3 text-sm text-neutral-400 mb-2">
        <Search className="h-4 w-4" />
        <span>Enter your research query to begin intelligent analysis</span>
      </div>

      <div
        className={`flex flex-row items-center justify-between text-white rounded-2xl ${
          hasHistory ? 'rounded-br-sm' : ''
        } break-words min-h-7 bg-neutral-700 px-6 py-4 border border-neutral-600`}
      >
        <Textarea
          value={internalInputValue}
          onChange={(e) => setInternalInputValue(e.target.value)}
          onKeyDown={handleInternalKeyDown}
          placeholder={placeholder || "What would you like to research? e.g., 'Latest developments in quantum computing and their commercial applications'"}
          className="w-full text-neutral-100 placeholder-neutral-500 resize-none border-0 focus:outline-none focus:ring-0 outline-none focus-visible:ring-0 shadow-none text-base min-h-[80px] max-h-[200px]"
          rows={3}
        />
        <div className="ml-4 flex-shrink-0">
          {isLoading ? (
            <Button
              type="button"
              variant="ghost"
              size="icon"
              className="text-red-500 hover:text-red-400 hover:bg-red-500/10 p-3 cursor-pointer rounded-full transition-all duration-200"
              onClick={onCancel}
            >
              <StopCircle className="h-6 w-6" />
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
            >
              <Send className="h-6 w-6" />
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
          >
            <SquarePen size={16} className="mr-2" />
            New Research
          </Button>
        </div>
      )}
    </form>
  );
};
