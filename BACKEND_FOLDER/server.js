import express from "express";
import dotenv from "dotenv";
import morgan from "morgan";
import cors from "cors";
import connectDB from "./config/db.js";

import stationRoutes from "./routes/stationRoutes.js";
import trainRoutes from "./routes/trainRoutes.js";
import timetableRoutes from "./routes/timetableRoutes.js";
import rollingStockRoutes from "./routes/rollingStockRoutes.js";
import eventRoutes from "./routes/eventRoutes.js";
import optimizationRoutes from "./routes/optimizationRoutes.js";
import kpiRoutes from "./routes/kpiRoutes.js";

dotenv.config();
connectDB();

const app = express();

// Middleware
app.use(cors());
app.use(express.json());
app.use(morgan("dev"));

// Routes
app.use("/api/stations", stationRoutes);
app.use("/api/trains", trainRoutes);
app.use("/api/timetables", timetableRoutes);
app.use("/api/rollingstock", rollingStockRoutes);
app.use("/api/events", eventRoutes);
app.use("/api/optimizations", optimizationRoutes);
app.use("/api/kpis", kpiRoutes);

// Server Start
const PORT = process.env.PORT || 5000;
app.listen(PORT, () => console.log(`🚆 Server running on port ${PORT}`));
