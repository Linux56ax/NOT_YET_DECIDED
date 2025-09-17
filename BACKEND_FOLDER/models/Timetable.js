import mongoose from "mongoose";

const timetableSchema = new mongoose.Schema({
  train_id: { type: mongoose.Schema.Types.ObjectId, ref: "Train" },
  station_id: { type: mongoose.Schema.Types.ObjectId, ref: "Station" },
  arrival_time: Date,
  departure_time: Date,
  platform: Number,
  status: { type: String, default: "on-time" }
});

export default mongoose.model("Timetable", timetableSchema);
