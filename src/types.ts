export interface Utterance {
  role: "agent" | "user";
  content: string;
}

// Events sent from Retell to your server
export interface CustomLlmRequest {
  interaction_type:
    | "call_details"
    | "ping_pong"
    | "update_only"
    | "response_required"
    | "reminder_required";
  // For call_details
  call?: Record<string, any>;
  // For ping_pong
  timestamp?: number;
  // For update_only, response_required, reminder_required
  transcript?: Utterance[];
  // For update_only
  turntaking?: "agent_turn" | "user_turn";
  // For response_required, reminder_required
  response_id?: number;
}

// Events sent from your server to Retell
export interface CustomLlmResponse {
  response_type:
    | "config"
    | "response"
    | "agent_interrupt"
    | "ping_pong"
    | "tool_call_invocation"
    | "tool_call_result";
  // For config
  config?: {
    auto_reconnect: boolean;
    call_details: boolean;
  };
  // For ping_pong
  timestamp?: number;
  // For response
  response_id?: number;
  content?: string;
  content_complete?: boolean;
  end_call?: boolean;
  no_interruption_allowed?: boolean;
  transfer_number?: string;
}
