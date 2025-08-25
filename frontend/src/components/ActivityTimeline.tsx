import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
} from '@/components/ui/card';
import { ScrollArea } from '@/components/ui/scroll-area';
import {
  Loader2,
  Activity,
  Info,
  Search,
  TextSearch,
  Brain,
  Pen,
  ChevronDown,
  ChevronUp,
} from 'lucide-react';
import { useEffect, useState } from 'react';

export interface ProcessedEvent {
  title: string;
  data: string | string[] | Record<string, unknown>;
  timestamp?: number; // Optional timestamp for duplicate detection and performance
}

interface ActivityTimelineProps {
  processedEvents: ProcessedEvent[];
  isLoading: boolean;
}

export function ActivityTimeline({
  processedEvents,
  isLoading,
}: ActivityTimelineProps) {
  const [isTimelineCollapsed, setIsTimelineCollapsed] =
    useState<boolean>(false);
  const getEventIcon = (title: string, index: number) => {
    if (index === 0 && isLoading && processedEvents.length === 0) {
      return <Loader2 className="h-4 w-4 text-neutral-400 animate-spin" />;
    }
    
    // Check if title already contains emoji, if so, don't add icon
    const hasEmoji = /[\u{1F600}-\u{1F6FF}\u{1F300}-\u{1F5FF}\u{1F900}-\u{1F9FF}\u{2600}-\u{26FF}\u{2700}-\u{27BF}]/u.test(title);
    if (hasEmoji) {
      return null; // No icon needed, emoji is in title
    }
    
    // Legacy icon mapping for titles without emojis
    const lowerTitle = title.toLowerCase();
    if (lowerTitle.includes('generating') || lowerTitle.includes('writing')) {
      return <TextSearch className="h-4 w-4 text-neutral-400" />;
    } else if (lowerTitle.includes('thinking') || lowerTitle.includes('processing')) {
      return <Loader2 className="h-4 w-4 text-neutral-400 animate-spin" />;
    } else if (lowerTitle.includes('reflection') || lowerTitle.includes('analyzing')) {
      return <Brain className="h-4 w-4 text-neutral-400" />;
    } else if (lowerTitle.includes('research') || lowerTitle.includes('search')) {
      return <Search className="h-4 w-4 text-neutral-400" />;
    } else if (lowerTitle.includes('finalizing') || lowerTitle.includes('completing')) {
      return <Pen className="h-4 w-4 text-neutral-400" />;
    }
    return <Activity className="h-4 w-4 text-neutral-400" />;
  };

  useEffect(() => {
    if (!isLoading && processedEvents.length !== 0) {
      setIsTimelineCollapsed(true);
    }
  }, [isLoading, processedEvents]);

  return (
    <Card className="border-none rounded-lg bg-neutral-700 w-full min-w-0 overflow-hidden">
      <CardHeader className="pb-3">
        <CardDescription className="flex items-center justify-between min-w-0">
          <button
            type="button"
            className="flex items-center justify-start text-sm w-full cursor-pointer gap-2 text-neutral-100 min-w-0 truncate bg-transparent border-none p-0 hover:text-neutral-50 focus:outline-none focus:ring-2 focus:ring-neutral-400 focus:ring-offset-2 focus:ring-offset-neutral-700 rounded"
            onClick={() => setIsTimelineCollapsed(!isTimelineCollapsed)}
            aria-expanded={!isTimelineCollapsed}
            aria-controls="timeline-content"
          >
            Research
            {isTimelineCollapsed ? (
              <ChevronDown className="h-4 w-4 mr-2 flex-shrink-0" />
            ) : (
              <ChevronUp className="h-4 w-4 mr-2 flex-shrink-0" />
            )}
          </button>
        </CardDescription>
      </CardHeader>
      {!isTimelineCollapsed && (
        <div className="max-h-80 overflow-hidden" id="timeline-content">
          <ScrollArea className="h-full max-h-80">
            <CardContent>
            {isLoading && processedEvents.length === 0 && (
              <div className="relative pl-8 pb-4 min-w-0">
                <div className="absolute left-3 top-3.5 h-full w-0.5 bg-neutral-800" />
                <div className="absolute left-0.5 top-2 h-5 w-5 rounded-full bg-neutral-800 flex items-center justify-center ring-4 ring-neutral-900">
                  <Loader2 className="h-3 w-3 text-neutral-400 animate-spin" />
                </div>
                <div className="min-w-0">
                  <p className="text-sm text-neutral-300 font-medium truncate">
                    Searching...
                  </p>
                </div>
              </div>
            )}
            {processedEvents.length > 0 ? (
              <div className="space-y-0 min-w-0">
                {processedEvents.map((eventItem, index) => (
                  <div key={index} className="relative pl-8 pb-4 min-w-0">
                    {index < processedEvents.length - 1 ||
                    (isLoading && index === processedEvents.length - 1) ? (
                      <div className="absolute left-3 top-3.5 h-full w-0.5 bg-neutral-600" />
                    ) : null}
                    <div className="absolute left-0.5 top-2 h-6 w-6 rounded-full bg-neutral-600 flex items-center justify-center ring-4 ring-neutral-700">
                      {getEventIcon(eventItem.title, index) || <Activity className="h-4 w-4 text-neutral-400" />}
                    </div>
                    <div className="min-w-0 flex-1">
                      <p className="text-sm text-neutral-200 font-medium mb-0.5 truncate">
                        {eventItem.title}
                      </p>
                      <p className="text-xs text-neutral-300 leading-relaxed break-words hyphens-auto max-w-full">
                        {typeof eventItem.data === 'string'
                          ? eventItem.data.length > 200 ? `${eventItem.data.substring(0, 200)}...` : eventItem.data
                          : Array.isArray(eventItem.data)
                          ? (eventItem.data as string[]).join(', ')
                          : JSON.stringify(eventItem.data)}
                      </p>
                    </div>
                  </div>
                ))}
                {isLoading && processedEvents.length > 0 && (
                  <div className="relative pl-8 pb-4 min-w-0">
                    <div className="absolute left-0.5 top-2 h-5 w-5 rounded-full bg-neutral-600 flex items-center justify-center ring-4 ring-neutral-700">
                      <Loader2 className="h-3 w-3 text-neutral-400 animate-spin" />
                    </div>
                    <div className="min-w-0">
                      <p className="text-sm text-neutral-300 font-medium truncate">
                        Searching...
                      </p>
                    </div>
                  </div>
                )}
              </div>
            ) : !isLoading ? ( // Only show "No activity" if not loading and no events
              <div className="flex flex-col items-center justify-center h-full text-neutral-500 pt-10 min-w-0">
                <Info className="h-6 w-6 mb-3 flex-shrink-0" />
                <p className="text-sm text-center">No activity to display.</p>
                <p className="text-xs text-neutral-600 mt-1 text-center">
                  Timeline will update during processing.
                </p>
              </div>
            ) : null}
            </CardContent>
          </ScrollArea>
        </div>
      )}
    </Card>
  );
}
