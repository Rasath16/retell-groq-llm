import express, { Request, Response } from "express";
import expressWs from "express-ws";
import { WebSocket } from "ws";
import { Retell } from "retell-sdk";
import { CustomLlmRequest, CustomLlmResponse } from "./types";
import { GroqLlmClient } from "./llm_groq";

export class Server {
  private app: expressWs.Application;
  private retellClient?: Retell;

  constructor() {
    const wsInstance = expressWs(express());
    this.app = wsInstance.app;

    this.app.use(express.json());
    this.app.use(express.urlencoded({ extended: true }));

    // Initialize Retell client if API key is provided (for webhook verification)
    if (process.env.RETELL_API_KEY) {
      this.retellClient = new Retell({ apiKey: process.env.RETELL_API_KEY });
    }

    this.setupRoutes();
  }

  private setupRoutes() {
    // Health check endpoint
    this.app.get("/", (req: Request, res: Response) => {
      res.json({ status: "ok", message: "Retell Groq Custom LLM Server" });
    });

    // Webhook endpoint for Retell events
    this.app.post("/webhook", (req: Request, res: Response) => {
      // Verify webhook signature if Retell API key is configured
      if (this.retellClient && req.headers["x-retell-signature"]) {
        const valid = Retell.verify(
          JSON.stringify(req.body),
          process.env.RETELL_API_KEY!,
          req.headers["x-retell-signature"] as string
        );
        if (!valid) {
          console.error("Invalid webhook signature");
          res.status(401).json({ error: "Invalid signature" });
          return;
        }
      }

      const event = req.body;
      switch (event.event) {
        case "call_started":
          console.log("Call started:", event.data?.call_id);
          break;
        case "call_ended":
          console.log("Call ended:", event.data?.call_id);
          break;
        case "call_analyzed":
          console.log("Call analyzed:", event.data?.call_id);
          break;
        default:
          console.log("Unknown webhook event:", event.event);
      }

      res.status(200).json({ received: true });
    });

    // WebSocket endpoint for Retell LLM communication
    this.app.ws(
      "/llm-websocket/:call_id",
      (ws: WebSocket, req: Request) => {
        const callId = req.params.call_id;
        console.log(`WebSocket connected for call: ${callId}`);

        const llmClient = new GroqLlmClient();

        // Send config event to Retell
        const configEvent: CustomLlmResponse = {
          response_type: "config",
          config: {
            auto_reconnect: true,
            call_details: true,
          },
        };
        ws.send(JSON.stringify(configEvent));

        ws.on("error", (error) => {
          console.error(`WebSocket error for call ${callId}:`, error);
        });

        ws.on("close", (code, reason) => {
          console.log(
            `WebSocket closed for call ${callId}: ${code} ${reason}`
          );
        });

        ws.on("message", async (data: string, isBinary: boolean) => {
          if (isBinary) {
            console.warn("Received binary message, ignoring");
            return;
          }

          try {
            const request: CustomLlmRequest = JSON.parse(data.toString());

            switch (request.interaction_type) {
              case "call_details":
                console.log(`Call details received for ${callId}`);
                // Agent speaks first — send the opening message
                llmClient.BeginMessage(ws);
                break;

              case "ping_pong":
                // Echo back ping with same timestamp
                const pong: CustomLlmResponse = {
                  response_type: "ping_pong",
                  timestamp: request.timestamp,
                };
                ws.send(JSON.stringify(pong));
                break;

              case "update_only":
                // Transcript update, no response needed
                break;

              case "response_required":
              case "reminder_required":
                // Generate and stream LLM response
                await llmClient.DraftResponse(request, ws);
                break;

              default:
                console.warn(
                  "Unknown interaction type:",
                  request.interaction_type
                );
            }
          } catch (error) {
            console.error("Error processing message:", error);
          }
        });
      }
    );
  }

  listen(port: number) {
    this.app.listen(port, () => {
      console.log(`Server running on port ${port}`);
      console.log(`WebSocket endpoint: ws://localhost:${port}/llm-websocket/{call_id}`);
    });
  }
}
