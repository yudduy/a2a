/**
 * Enhanced Error Boundary with production-ready error handling
 * 
 * Features:
 * - Error recovery mechanisms
 * - Detailed error logging
 * - User-friendly error display
 * - Automatic retry functionality
 * - Performance monitoring integration
 */

import React, { Component, ErrorInfo, ReactNode, PropsWithChildren } from 'react';
import { AlertTriangle, RefreshCw, Home, Copy, CopyCheck } from 'lucide-react';
import { Button } from './button';
import { Card, CardContent, CardHeader, CardTitle } from './card';

interface ErrorBoundaryState {
  hasError: boolean;
  error: Error | null;
  errorInfo: ErrorInfo | null;
  retryCount: number;
  showDetails: boolean;
  copiedError: boolean;
}

interface ErrorBoundaryProps {
  fallback?: ReactNode;
  onError?: (error: Error, errorInfo: ErrorInfo) => void;
  enableRetry?: boolean;
  maxRetries?: number;
  resetOnPropsChange?: boolean;
  resetKeys?: Array<string | number>;
  level?: 'page' | 'component' | 'feature';
}

type ErrorBoundaryPropsWithChildren = PropsWithChildren<ErrorBoundaryProps>;

export class EnhancedErrorBoundary extends Component<
  ErrorBoundaryPropsWithChildren,
  ErrorBoundaryState
> {
  private retryTimeoutId: number | null = null;

  constructor(props: ErrorBoundaryPropsWithChildren) {
    super(props);
    
    this.state = {
      hasError: false,
      error: null,
      errorInfo: null,
      retryCount: 0,
      showDetails: false,
      copiedError: false,
    };
  }

  static getDerivedStateFromError(error: Error): Partial<ErrorBoundaryState> {
    return {
      hasError: true,
      error,
    };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error('Error caught by boundary:', error, errorInfo);
    
    this.setState({
      error,
      errorInfo,
    });

    // Call optional error handler
    this.props.onError?.(error, errorInfo);

    // Log to external error reporting service
    this.logError(error, errorInfo);
  }

  componentDidUpdate(prevProps: ErrorBoundaryPropsWithChildren) {
    const { resetOnPropsChange, resetKeys } = this.props;
    const { hasError } = this.state;

    // Reset error state when specified props change
    if (hasError && resetOnPropsChange && resetKeys) {
      const hasResetKeyChanged = resetKeys.some(
        (resetKey, idx) => prevProps.resetKeys?.[idx] !== resetKey
      );

      if (hasResetKeyChanged) {
        this.resetErrorBoundary();
      }
    }
  }

  componentWillUnmount() {
    if (this.retryTimeoutId) {
      clearTimeout(this.retryTimeoutId);
    }
  }

  private logError = (error: Error, errorInfo: ErrorInfo) => {
    // In production, send to error reporting service
    const errorReport = {
      message: error.message,
      stack: error.stack,
      componentStack: errorInfo.componentStack,
      timestamp: new Date().toISOString(),
      userAgent: navigator.userAgent,
      url: window.location.href,
      level: this.props.level || 'component',
    };

    // For development, just log to console
    if (process.env.NODE_ENV === 'development') {
      console.error('Error Report:', errorReport);
    } else {
      // In production, send to monitoring service
      // Example: Sentry, LogRocket, etc.
      this.sendErrorReport(errorReport);
    }
  };

  private sendErrorReport = async (errorReport: any) => {
    try {
      // Example implementation - replace with your error reporting service
      await fetch('/api/errors', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(errorReport),
      });
    } catch (reportingError) {
      console.error('Failed to send error report:', reportingError);
    }
  };

  private resetErrorBoundary = () => {
    this.setState({
      hasError: false,
      error: null,
      errorInfo: null,
      retryCount: 0,
      showDetails: false,
      copiedError: false,
    });
  };

  private handleRetry = () => {
    const { maxRetries = 3 } = this.props;
    const { retryCount } = this.state;

    if (retryCount < maxRetries) {
      this.setState(
        prevState => ({ retryCount: prevState.retryCount + 1 }),
        () => {
          // Add a small delay before retrying
          this.retryTimeoutId = window.setTimeout(() => {
            this.resetErrorBoundary();
          }, 1000);
        }
      );
    }
  };

  private toggleDetails = () => {
    this.setState(prevState => ({
      showDetails: !prevState.showDetails,
    }));
  };

  private copyErrorToClipboard = async () => {
    const { error, errorInfo } = this.state;
    
    if (!error || !errorInfo) return;

    const errorText = `
Error: ${error.message}

Stack Trace:
${error.stack}

Component Stack:
${errorInfo.componentStack}

Timestamp: ${new Date().toISOString()}
URL: ${window.location.href}
User Agent: ${navigator.userAgent}
    `.trim();

    try {
      await navigator.clipboard.writeText(errorText);
      this.setState({ copiedError: true });
      setTimeout(() => this.setState({ copiedError: false }), 2000);
    } catch (err) {
      console.error('Failed to copy error to clipboard:', err);
    }
  };

  private renderFallbackUI() {
    const { error, errorInfo, retryCount, showDetails, copiedError } = this.state;
    const { enableRetry = true, maxRetries = 3, level = 'component' } = this.props;

    const canRetry = enableRetry && retryCount < maxRetries;
    const isPageLevel = level === 'page';

    return (
      <Card className="border-red-500/30 bg-red-500/5 max-w-2xl mx-auto">
        <CardHeader>
          <div className="flex items-center gap-3">
            <div className="h-10 w-10 rounded-full bg-red-500/20 flex items-center justify-center">
              <AlertTriangle className="h-5 w-5 text-red-400" />
            </div>
            <div>
              <CardTitle className="text-red-200">
                {isPageLevel ? 'Page Error' : 'Component Error'}
              </CardTitle>
              <p className="text-sm text-red-300 mt-1">
                Something went wrong while {isPageLevel ? 'loading this page' : 'rendering this component'}
              </p>
            </div>
          </div>
        </CardHeader>

        <CardContent className="space-y-4">
          <div className="text-sm text-red-200">
            <p className="font-medium mb-2">Error Message:</p>
            <div className="bg-red-500/10 border border-red-500/30 rounded p-3 font-mono text-xs">
              {error?.message || 'Unknown error occurred'}
            </div>
          </div>

          <div className="flex flex-wrap gap-2">
            {canRetry && (
              <Button
                onClick={this.handleRetry}
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
              className="border-red-500/30 text-red-300 hover:bg-red-500/10"
            >
              <Home className="h-4 w-4 mr-2" />
              Reset
            </Button>

            <Button
              onClick={this.toggleDetails}
              variant="ghost"
              size="sm"
              className="text-red-300 hover:bg-red-500/10"
            >
              {showDetails ? 'Hide' : 'Show'} Details
            </Button>

            <Button
              onClick={this.copyErrorToClipboard}
              variant="ghost"
              size="sm"
              className="text-red-300 hover:bg-red-500/10"
            >
              {copiedError ? (
                <CopyCheck className="h-4 w-4 mr-2" />
              ) : (
                <Copy className="h-4 w-4 mr-2" />
              )}
              {copiedError ? 'Copied!' : 'Copy Error'}
            </Button>
          </div>

          {showDetails && (
            <div className="space-y-3 text-xs">
              <div>
                <p className="font-medium text-red-200 mb-2">Stack Trace:</p>
                <div className="bg-red-500/10 border border-red-500/30 rounded p-3 font-mono overflow-auto max-h-40">
                  <pre className="whitespace-pre-wrap text-red-100">
                    {error?.stack || 'No stack trace available'}
                  </pre>
                </div>
              </div>

              {errorInfo?.componentStack && (
                <div>
                  <p className="font-medium text-red-200 mb-2">Component Stack:</p>
                  <div className="bg-red-500/10 border border-red-500/30 rounded p-3 font-mono overflow-auto max-h-32">
                    <pre className="whitespace-pre-wrap text-red-100">
                      {errorInfo.componentStack}
                    </pre>
                  </div>
                </div>
              )}
            </div>
          )}

          <div className="text-xs text-red-300 text-center pt-2 border-t border-red-500/30">
            If this error persists, please contact support with the error details above.
          </div>
        </CardContent>
      </Card>
    );
  }

  render() {
    if (this.state.hasError) {
      return this.props.fallback || this.renderFallbackUI();
    }

    return this.props.children;
  }
}

// Higher-order component for easier usage
export function withErrorBoundary<P extends object>(
  Component: React.ComponentType<P>,
  errorBoundaryProps?: ErrorBoundaryProps
) {
  const WrappedComponent = (props: P) => (
    <EnhancedErrorBoundary {...errorBoundaryProps}>
      <Component {...props} />
    </EnhancedErrorBoundary>
  );

  WrappedComponent.displayName = `withErrorBoundary(${Component.displayName || Component.name})`;
  
  return WrappedComponent;
}

// Hook for manual error boundary triggering
export function useErrorHandler() {
  const [, setState] = React.useState();
  
  return React.useCallback((error: Error) => {
    setState(() => {
      throw error;
    });
  }, []);
}

export default EnhancedErrorBoundary;