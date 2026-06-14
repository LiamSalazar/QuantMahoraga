import { Component, type ErrorInfo, type ReactNode } from "react";
import { ErrorState } from "./States";

type State = { error: string | null };

export class ErrorBoundary extends Component<{ children: ReactNode; resetKey?: string }, State> {
  state: State = { error: null };

  static getDerivedStateFromError(error: Error): State {
    return { error: error.message };
  }

  componentDidUpdate(prevProps: { resetKey?: string }) {
    if (prevProps.resetKey !== this.props.resetKey && this.state.error) {
      this.setState({ error: null });
    }
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error("Mahoraga DSS view error", error, info);
  }

  render() {
    if (this.state.error) return <ErrorState error={this.state.error} retry={() => this.setState({ error: null })} />;
    return this.props.children;
  }
}
