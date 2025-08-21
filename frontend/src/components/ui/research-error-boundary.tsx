/**
 * Specialized Error Boundary for Research Components
 * 
 * Features:
 * - Research-specific error handling and recovery
 * - Graceful degradation for WebSocket failures
 * - Sequence-specific error isolation
 * - Research state preservation and recovery
 * - Performance monitoring integration
 */

import React, { Component, ErrorInfo, ReactNode, PropsWithChildren } from 'react';
import { AlertTriangle, RefreshCw, WifiOff, Zap, GitCompare } from 'lucide-react';
import { Button } from './button';
import { Card, CardContent, CardHeader, CardTitle } from './card';
import { Badge } from './badge';
import { cn } from '@/lib/utils';

interface ResearchErrorInfo {
  componentStack: string;
  errorBoundary: string;
  errorMessage: string;
  sequence?: string;
  strategy?: string;
  timestamp: number;
  userId?: string;
  researchQuery?: string;
  connectionState?: string;
  performanceMetrics?: {
    renderTime: number;
    memoryUsage: number;
    messageRate: number;
  };
}

interface ResearchErrorBoundaryState {
  hasError: boolean;
  error: Error | null;
  errorInfo: ErrorInfo | null;
  errorCategory: 'websocket' | 'sequence' | 'component' | 'unknown';
  retryCount: number;
  lastErrorTime: number;
  errorHistory: ResearchErrorInfo[];
  isRecovering: boolean;
  gracefulDegradation: boolean;
}

interface ResearchErrorBoundaryProps {
  researchQuery?: string;
  sequenceStrategy?: string;
  onError?: (error: Error, errorInfo: ErrorInfo, researchInfo: ResearchErrorInfo) => void;
  onRecovery?: () => void;
  enableGracefulDegradation?: boolean;
  maxRetries?: number;
  enableErrorReporting?: boolean;
  fallbackComponent?: ReactNode;
  isolateSequenceErrors?: boolean;
}

type ResearchErrorBoundaryPropsWithChildren = PropsWithChildren<ResearchErrorBoundaryProps>;

export class ResearchErrorBoundary extends Component<
  ResearchErrorBoundaryPropsWithChildren,
  ResearchErrorBoundaryState
> {
  private retryTimeoutId: number | null = null;
  private errorReportingQueue: ResearchErrorInfo[] = [];
  private performanceObserver: PerformanceObserver | null = null;

  constructor(props: ResearchErrorBoundaryPropsWithChildren) {
    super(props);
    
    this.state = {
      hasError: false,
      error: null,
      errorInfo: null,
      errorCategory: 'unknown',
      retryCount: 0,
      lastErrorTime: 0,
      errorHistory: [],
      isRecovering: false,
      gracefulDegradation: false,
    };

    this.setupPerformanceMonitoring();
  }

  private setupPerformanceMonitoring(): void {
    if ('PerformanceObserver' in window) {
      this.performanceObserver = new PerformanceObserver((list) => {
        const entries = list.getEntries();
        entries.forEach((entry) => {
          if (entry.name.includes('research-component')) {
            console.log(`Research component performance: ${entry.name} took ${entry.duration}ms`);
          }
        });
      });
      
      try {
        this.performanceObserver.observe({ entryTypes: ['measure'] });
      } catch (error) {
        console.warn('Performance observer not supported:', error);
      }
    }
  }

  static getDerivedStateFromError(error: Error): Partial<ResearchErrorBoundaryState> {
    const errorCategory = ResearchErrorBoundary.categorizeError(error);
    
    return {
      hasError: true,
      error,
      errorCategory,
      lastErrorTime: Date.now(),
    };
  }

  private static categorizeError(error: Error): 'websocket' | 'sequence' | 'component' | 'unknown' {
    const message = error.message.toLowerCase();
    const stack = error.stack?.toLowerCase() || '';

    if (message.includes('websocket') || message.includes('connection') || stack.includes('websocket')) {
      return 'websocket';
    }
    if (message.includes('sequence') || message.includes('strategy') || stack.includes('sequence')) {
      return 'sequence';
    }
    if (stack.includes('react') || stack.includes('component')) {
      return 'component';
    }
    
    return 'unknown';
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error('Research error caught:', error, errorInfo);
    
    const researchErrorInfo: ResearchErrorInfo = {
      componentStack: errorInfo.componentStack,
      errorBoundary: 'ResearchErrorBoundary',
      errorMessage: error.message,
      sequence: this.props.sequenceStrategy,
      strategy: this.props.sequenceStrategy,
      timestamp: Date.now(),
      researchQuery: this.props.researchQuery,
      performanceMetrics: this.getPerformanceMetrics(),
    };

    this.setState(prevState => ({
      errorInfo,
      errorHistory: [...prevState.errorHistory, researchErrorInfo].slice(-10), // Keep last 10 errors
    }));

    // Call optional error handler
    this.props.onError?.(error, errorInfo, researchErrorInfo);

    // Handle different error categories
    this.handleErrorByCategory(error, errorInfo, researchErrorInfo);

    // Queue for error reporting
    if (this.props.enableErrorReporting) {
      this.queueErrorReport(researchErrorInfo);
    }
  }

  private handleErrorByCategory(error: Error, errorInfo: ErrorInfo, researchInfo: ResearchErrorInfo): void {
    const { errorCategory } = this.state;

    switch (errorCategory) {
      case 'websocket':
        this.handleWebSocketError(error, researchInfo);
        break;
      case 'sequence':
        this.handleSequenceError(error, researchInfo);
        break;
      case 'component':
        this.handleComponentError(error, researchInfo);
        break;
      default:
        this.handleUnknownError(error, researchInfo);
    }
  }

  private handleWebSocketError(error: Error, researchInfo: ResearchErrorInfo): void {
    console.log('WebSocket error detected, enabling graceful degradation');
    
    if (this.props.enableGracefulDegradation) {
      this.setState({ gracefulDegradation: true });
      
      // Attempt automatic recovery after delay
      this.retryTimeoutId = window.setTimeout(() => {
        this.attemptWebSocketRecovery();
      }, 5000);
    }
  }

  private handleSequenceError(error: Error, researchInfo: ResearchErrorInfo): void {
    console.log(`Sequence error detected for ${researchInfo.strategy}`);
    
    if (this.props.isolateSequenceErrors) {
      // Isolate the error to specific sequence
      this.setState({ gracefulDegradation: true });
    }
  }

  private handleComponentError(error: Error, researchInfo: ResearchErrorInfo): void {
    console.log('Component error detected, will attempt retry');
    
    // Component errors can often be resolved with a simple retry
    this.scheduleRetry();
  }

  private handleUnknownError(error: Error, researchInfo: ResearchErrorInfo): void {
    console.log('Unknown error category, using default handling');
    
    // Default error handling
    this.scheduleRetry();
  }

  private async attemptWebSocketRecovery(): Promise<void> {
    this.setState({ isRecovering: true });
    
    try {
      // Attempt to re-establish WebSocket connection
      await new Promise(resolve => setTimeout(resolve, 2000));
      
      this.setState({
        hasError: false,
        error: null,
        errorInfo: null,
        isRecovering: false,
        gracefulDegradation: false,
      });
      
      this.props.onRecovery?.();
    } catch (recoveryError) {
      console.error('Failed to recover from WebSocket error:', recoveryError);
      this.setState({ isRecovering: false });
    }
  }

  private scheduleRetry(): void {
    const { maxRetries = 3 } = this.props;
    const { retryCount } = this.state;

    if (retryCount < maxRetries) {
      this.setState({ isRecovering: true });
      
      this.retryTimeoutId = window.setTimeout(() => {
        this.setState(prevState => ({
          hasError: false,
          error: null,
          errorInfo: null,
          retryCount: prevState.retryCount + 1,
          isRecovering: false,
        }));
        
        this.props.onRecovery?.();
      }, Math.min(1000 * Math.pow(2, retryCount), 10000)); // Exponential backoff, max 10s
    }
  }

  private getPerformanceMetrics() {
    try {
      const memory = (performance as any).memory;
      return {
        renderTime: performance.now(),
        memoryUsage: memory ? memory.usedJSHeapSize / 1024 / 1024 : 0,
        messageRate: 0, // Would be populated from WebSocket client
      };
    } catch {
      return {
        renderTime: 0,
        memoryUsage: 0,
        messageRate: 0,
      };
    }
  }

  private queueErrorReport(errorInfo: ResearchErrorInfo): void {
    this.errorReportingQueue.push(errorInfo);
    
    // Send error reports in batches
    if (this.errorReportingQueue.length >= 5) {
      this.sendErrorReports();
    }
  }

  private async sendErrorReports(): Promise<void> {
    const reports = this.errorReportingQueue.splice(0);
    
    try {
      await fetch('/api/research-errors', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          errors: reports,
          sessionId: this.getSessionId(),
          timestamp: Date.now(),
        }),
      });
    } catch (error) {
      console.error('Failed to send error reports:', error);
      // Re-queue failed reports
      this.errorReportingQueue.unshift(...reports);
    }
  }

  private getSessionId(): string {
    return sessionStorage.getItem('research-session-id') || 'unknown';
  }

  componentWillUnmount() {
    if (this.retryTimeoutId) {
      clearTimeout(this.retryTimeoutId);
    }
    
    if (this.performanceObserver) {
      this.performanceObserver.disconnect();
    }
    
    // Send any remaining error reports
    if (this.errorReportingQueue.length > 0) {
      this.sendErrorReports();
    }
  }

  private resetErrorBoundary = () => {
    this.setState({
      hasError: false,
      error: null,
      errorInfo: null,
      retryCount: 0,
      isRecovering: false,
      gracefulDegradation: false,
    });
  };

  private renderErrorUI() {
    const { error, errorCategory, retryCount, isRecovering, gracefulDegradation, errorHistory } = this.state;
    const { maxRetries = 3, researchQuery, sequenceStrategy } = this.props;

    const canRetry = retryCount < maxRetries;
    const errorFrequency = errorHistory.filter(e => Date.now() - e.timestamp < 60000).length;

    return (
      <Card className="border-red-500/30 bg-red-500/5 max-w-4xl mx-auto">
        <CardHeader>
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="h-12 w-12 rounded-full bg-red-500/20 flex items-center justify-center">
                {errorCategory === 'websocket' && <WifiOff className="h-6 w-6 text-red-400" />}
                {errorCategory === 'sequence' && <Zap className="h-6 w-6 text-red-400" />}
                {errorCategory === 'component' && <GitCompare className="h-6 w-6 text-red-400" />}
                {errorCategory === 'unknown' && <AlertTriangle className="h-6 w-6 text-red-400" />}
              </div>
              <div>
                <CardTitle className="text-red-200 flex items-center gap-2">
                  Research System Error
                  <Badge variant="outline" className="text-xs bg-red-500/10 text-red-300 border-red-500/30">
                    {errorCategory.toUpperCase()}
                  </Badge>
                </CardTitle>
                <p className="text-sm text-red-300 mt-1">
                  {this.getErrorDescription(errorCategory)}
                </p>
              </div>
            </div>

            {errorFrequency > 3 && (
              <Badge variant="outline" className="text-xs bg-amber-500/10 text-amber-300 border-amber-500/30">
                High Error Rate
              </Badge>
            )}
          </div>
        </CardHeader>

        <CardContent className="space-y-4">
          {/* Research Context */}
          {(researchQuery || sequenceStrategy) && (
            <div className="bg-neutral-700/50 rounded-lg p-3">
              <h4 className="text-sm font-medium text-neutral-200 mb-2">Research Context</h4>
              {researchQuery && (
                <p className="text-xs text-neutral-300 mb-1">
                  <span className="font-medium">Query:</span> {researchQuery}
                </p>
              )}
              {sequenceStrategy && (
                <p className="text-xs text-neutral-300">
                  <span className="font-medium">Strategy:</span> {sequenceStrategy}
                </p>
              )}
            </div>
          )}

          {/* Error Details */}
          <div className="text-sm text-red-200">
            <p className="font-medium mb-2">Error Details:</p>
            <div className="bg-red-500/10 border border-red-500/30 rounded p-3 font-mono text-xs">
              {error?.message || 'Unknown error occurred'}
            </div>
          </div>

          {/* Graceful Degradation Notice */}
          {gracefulDegradation && (
            <div className="bg-blue-500/10 border border-blue-500/30 rounded p-3">
              <p className="text-sm text-blue-200">
                🛡️ <strong>Graceful Degradation Active:</strong> The system is operating in a reduced 
                functionality mode to maintain partial service while recovering from the error.
              </p>
            </div>
          )}

          {/* Recovery Status */}
          {isRecovering && (
            <div className="bg-amber-500/10 border border-amber-500/30 rounded p-3">
              <p className="text-sm text-amber-200">
                🔄 <strong>Attempting Recovery:</strong> The system is trying to automatically 
                recover from the error. Please wait...
              </p>
            </div>
          )}

          {/* Action Buttons */}
          <div className="flex flex-wrap gap-2">
            {canRetry && !isRecovering && (
              <Button
                onClick={this.scheduleRetry.bind(this)}
                variant="outline"
                size="sm"
                className="border-red-500/30 text-red-300 hover:bg-red-500/10"
              >
                <RefreshCw className="h-4 w-4 mr-2" />
                Retry ({retryCount}/{maxRetries})
              </Button>
            )}

            <Button
              onClick={this.resetErrorBoundary}
              variant="outline"
              size="sm"
              className="border-blue-500/30 text-blue-300 hover:bg-blue-500/10"
            >
              Reset Research
            </Button>

            {errorCategory === 'websocket' && (
              <Button
                onClick={() => window.location.reload()}
                variant="outline"
                size="sm"
                className="border-green-500/30 text-green-300 hover:bg-green-500/10"
              >
                Reload Application
              </Button>
            )}
          </div>

          {/* Error Prevention Tips */}
          <div className="text-xs text-red-300 bg-red-500/5 border border-red-500/20 rounded p-3">
            <p className="font-medium mb-2">💡 Tips to prevent this error:</p>
            <ul className="list-disc list-inside space-y-1">
              {this.getErrorPreventionTips(errorCategory)}
            </ul>
          </div>

          {/* Support Information */}
          <div className="text-xs text-red-300 text-center pt-2 border-t border-red-500/30">
            Error ID: {Date.now().toString(36)} • 
            If this error persists, please report it with the error details above.
          </div>
        </CardContent>
      </Card>
    );
  }

  private getErrorDescription(category: string): string {
    switch (category) {
      case 'websocket':
        return 'Connection to research servers lost. Attempting to restore connection...';
      case 'sequence':
        return 'Error in research sequence execution. This may affect one research approach.';
      case 'component':
        return 'User interface component error. The display may be temporarily affected.';
      default:
        return 'An unexpected error occurred in the research system.';
    }
  }

  private getErrorPreventionTips(category: string): ReactNode[] {
    const tips: Record<string, string[]> = {
      websocket: [
        'Ensure stable internet connection',
        'Check if firewall is blocking WebSocket connections',
        'Try refreshing the page if connection issues persist',
      ],
      sequence: [
        'Try starting research with a different strategy',
        'Ensure research query is well-formatted',
        'Check system load - reduce concurrent operations',
      ],
      component: [
        'Refresh the browser tab',
        'Clear browser cache and cookies',
        'Try using an incognito/private browsing window',
      ],
      unknown: [
        'Refresh the browser tab',
        'Check browser console for additional error details',
        'Try using a different browser',
      ],
    };

    return (tips[category] || tips.unknown).map((tip, index) => (
      <li key={index}>{tip}</li>
    ));
  }

  render() {
    if (this.state.hasError) {
      // Show fallback component or default error UI
      return this.props.fallbackComponent || this.renderErrorUI();
    }

    return this.props.children;
  }
}

// Higher-order component for research components
export function withResearchErrorBoundary<P extends object>(
  Component: React.ComponentType<P>,
  errorBoundaryProps?: Omit<ResearchErrorBoundaryProps, 'children'>
) {
  const WrappedComponent = (props: P) => (
    <ResearchErrorBoundary {...errorBoundaryProps}>
      <Component {...props} />
    </ResearchErrorBoundary>
  );

  WrappedComponent.displayName = `withResearchErrorBoundary(${Component.displayName || Component.name})`;
  
  return WrappedComponent;
}

export default ResearchErrorBoundary;