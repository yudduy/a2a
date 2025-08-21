/**
 * Production Error Boundary Component
 * 
 * Provides comprehensive error handling for the parallel research interface,
 * with graceful degradation and recovery options.
 */

import React, { Component, ErrorInfo, ReactNode } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { 
  AlertTriangle, 
  RefreshCw, 
  Home, 
  Bug, 
  ExternalLink,
  ChevronDown,
  ChevronUp
} from 'lucide-react';

interface Props {
  children: ReactNode;
  fallback?: ReactNode;
  onError?: (error: Error, errorInfo: ErrorInfo) => void;
  resetOnPropsChange?: boolean;
  isolateComponent?: boolean;
}

interface State {
  hasError: boolean;
  error: Error | null;
  errorInfo: ErrorInfo | null;
  showDetails: boolean;
  errorId: string;
}

export class ErrorBoundary extends Component<Props, State> {
  private resetTimeoutId: number | null = null;

  public state: State = {
    hasError: false,
    error: null,
    errorInfo: null,
    showDetails: false,
    errorId: '',
  };

  constructor(props: Props) {
    super(props);
    this.handleReset = this.handleReset.bind(this);
    this.handleReportError = this.handleReportError.bind(this);
    this.toggleDetails = this.toggleDetails.bind(this);
  }

  public static getDerivedStateFromError(error: Error): Partial<State> {
    // Update state so the next render will show the fallback UI
    return {
      hasError: true,
      error,
      errorId: `error_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
    };
  }

  public componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    // Log error details
    console.error('ErrorBoundary caught an error:', error, errorInfo);
    
    this.setState({
      error,
      errorInfo,
    });

    // Call custom error handler if provided
    if (this.props.onError) {
      this.props.onError(error, errorInfo);
    }

    // Auto-recovery attempt for transient errors
    if (this.isTransientError(error)) {
      this.scheduleAutoRecovery();
    }
  }

  public componentDidUpdate(prevProps: Props) {
    const { resetOnPropsChange, children } = this.props;
    const { hasError } = this.state;
    
    // Reset error state when children change (if enabled)
    if (hasError && resetOnPropsChange && prevProps.children !== children) {
      this.handleReset();
    }
  }

  public componentWillUnmount() {
    if (this.resetTimeoutId) {
      clearTimeout(this.resetTimeoutId);
    }
  }

  private isTransientError(error: Error): boolean {
    // Check for network, WebSocket, or other recoverable errors
    const transientPatterns = [
      /network/i,
      /websocket/i,
      /connection/i,
      /timeout/i,
      /loading/i,
      /fetch/i,
    ];
    
    return transientPatterns.some(pattern => 
      pattern.test(error.message) || pattern.test(error.name)
    );
  }

  private scheduleAutoRecovery() {
    // Attempt recovery after 5 seconds for transient errors
    this.resetTimeoutId = window.setTimeout(() => {
      console.log('Attempting automatic error recovery...');
      this.handleReset();
    }, 5000);
  }

  private handleReset() {
    if (this.resetTimeoutId) {
      clearTimeout(this.resetTimeoutId);
      this.resetTimeoutId = null;
    }
    
    this.setState({
      hasError: false,
      error: null,
      errorInfo: null,
      showDetails: false,
      errorId: '',
    });
  }

  private handleReportError() {
    const { error, errorInfo, errorId } = this.state;
    
    // Create error report
    const errorReport = {
      errorId,
      timestamp: new Date().toISOString(),
      userAgent: navigator.userAgent,
      url: window.location.href,
      error: {
        name: error?.name,
        message: error?.message,
        stack: error?.stack,
      },
      errorInfo: {
        componentStack: errorInfo?.componentStack,
      },
      appState: {
        // Add any relevant app state here
      },
    };

    console.log('Error Report:', errorReport);
    
    // In production, you would send this to your error reporting service
    // Example: errorReportingService.send(errorReport);
    
    // For now, copy to clipboard for user to report
    if (navigator.clipboard) {
      navigator.clipboard.writeText(JSON.stringify(errorReport, null, 2))
        .then(() => {
          alert('Error details copied to clipboard. Please report this issue.');
        })
        .catch(() => {
          console.error('Failed to copy error report to clipboard');
        });
    }
  }

  private toggleDetails() {
    this.setState(prev => ({ showDetails: !prev.showDetails }));
  }

  public render() {
    if (this.state.hasError) {
      // Custom fallback UI
      if (this.props.fallback) {
        return this.props.fallback;
      }

      const { error, errorInfo, showDetails, errorId } = this.state;
      const isTransient = error ? this.isTransientError(error) : false;

      return (
        <div className="min-h-[400px] flex items-center justify-center p-6 bg-neutral-900">
          <Card className="max-w-2xl w-full border-red-500/30 bg-neutral-800">
            <CardHeader>
              <div className="flex items-center gap-3">
                <div className="h-10 w-10 rounded-lg bg-red-500/20 flex items-center justify-center">
                  <AlertTriangle className="h-5 w-5 text-red-400" />
                </div>
                <div>
                  <CardTitle className="text-red-100">
                    Something went wrong
                  </CardTitle>
                  <div className="flex items-center gap-2 mt-1">
                    <Badge variant="outline" className="text-xs bg-red-500/10 text-red-300 border-red-500/30">
                      Error ID: {errorId.slice(-8)}
                    </Badge>
                    {isTransient && (
                      <Badge variant="outline" className="text-xs bg-amber-500/10 text-amber-300 border-amber-500/30">
                        Auto-recovering...
                      </Badge>
                    )}
                  </div>
                </div>
              </div>
            </CardHeader>
            
            <CardContent className="space-y-6">
              <div>
                <p className="text-neutral-300 leading-relaxed">
                  We encountered an unexpected error in the parallel research interface. 
                  {isTransient 
                    ? ' This appears to be a temporary issue that we\'re attempting to recover from automatically.' 
                    : ' Please try the recovery options below or start a new session.'}
                </p>
              </div>

              {/* Error Details (collapsible) */}
              {error && (
                <div className="border border-neutral-600 rounded-lg">
                  <button
                    onClick={this.toggleDetails}
                    className="w-full flex items-center justify-between p-3 text-left hover:bg-neutral-700/50 transition-colors"
                  >
                    <div className="flex items-center gap-2">
                      <Bug className="h-4 w-4 text-neutral-400" />
                      <span className="text-sm font-medium text-neutral-200">
                        Technical Details
                      </span>
                    </div>
                    {showDetails ? (
                      <ChevronUp className="h-4 w-4 text-neutral-400" />
                    ) : (
                      <ChevronDown className="h-4 w-4 text-neutral-400" />
                    )}
                  </button>
                  
                  {showDetails && (
                    <div className="border-t border-neutral-600 p-3 bg-neutral-900/50">
                      <div className="space-y-2">
                        <div>
                          <p className="text-xs font-medium text-neutral-300 mb-1">
                            Error Message:
                          </p>
                          <p className="text-xs text-red-300 font-mono bg-neutral-800 p-2 rounded">
                            {error.message}
                          </p>
                        </div>
                        
                        {error.stack && (
                          <div>
                            <p className="text-xs font-medium text-neutral-300 mb-1">
                              Stack Trace:
                            </p>
                            <pre className="text-xs text-neutral-400 bg-neutral-800 p-2 rounded max-h-40 overflow-y-auto font-mono">
                              {error.stack}
                            </pre>
                          </div>
                        )}
                        
                        {errorInfo?.componentStack && (
                          <div>
                            <p className="text-xs font-medium text-neutral-300 mb-1">
                              Component Stack:
                            </p>
                            <pre className="text-xs text-neutral-400 bg-neutral-800 p-2 rounded max-h-32 overflow-y-auto font-mono">
                              {errorInfo.componentStack}
                            </pre>
                          </div>
                        )}
                      </div>
                    </div>
                  )}
                </div>
              )}

              {/* Recovery Actions */}
              <div className="flex flex-col sm:flex-row gap-3">
                <Button
                  onClick={this.handleReset}
                  className="flex-1 bg-blue-600 hover:bg-blue-700 text-white"
                >
                  <RefreshCw className="h-4 w-4 mr-2" />
                  Try Again
                </Button>
                
                <Button
                  onClick={() => window.location.href = '/'}
                  variant="outline"
                  className="flex-1 border-neutral-600 text-neutral-300 hover:bg-neutral-700"
                >
                  <Home className="h-4 w-4 mr-2" />
                  New Session
                </Button>
                
                <Button
                  onClick={this.handleReportError}
                  variant="outline"
                  className="flex-1 border-amber-600 text-amber-400 hover:bg-amber-600/10"
                >
                  <ExternalLink className="h-4 w-4 mr-2" />
                  Report Issue
                </Button>
              </div>

              {/* Help Text */}
              <div className="text-xs text-neutral-500 text-center">
                If this problem persists, please refresh the page or contact support.
                {isTransient && ' Automatic recovery will be attempted in a few seconds.'}
              </div>
            </CardContent>
          </Card>
        </div>
      );
    }

    return this.props.children;
  }
}

export default ErrorBoundary;