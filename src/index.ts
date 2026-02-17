import dotenv from "dotenv";

// Load environment variables
const envFile =
  process.env.NODE_ENV === "development" ? ".env.development" : ".env";
dotenv.config({ path: envFile });

import { Server } from "./server";

const server = new Server();
const port = parseInt(process.env.PORT || "8080", 10);
server.listen(port);
