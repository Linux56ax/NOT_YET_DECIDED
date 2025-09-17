import express from "express";
import Timetable from "../models/Timetable.js";

const router = express.Router();

// get all timetables
router.get("/", async (req, res) => {
  const timetables = await Timetable.find().populate("train_id station_id");
  res.json(timetables);
});

// get timetable by ID
router.get("/:id", async (req, res) => {
  const timetable = await Timetable.findById(req.params.id).populate("train_id station_id");
  res.json(timetable);
});

// post new timetable
router.post("/", async (req, res) => {
  const timetable = new Timetable(req.body);
  await timetable.save();
  res.json(timetable);
});

// put update timetable
router.put("/:id", async (req, res) => {
  const timetable = await Timetable.findByIdAndUpdate(req.params.id, req.body, { new: true });
  res.json(timetable);
});

// delete timetable
router.delete("/:id", async (req, res) => {
  await Timetable.findByIdAndDelete(req.params.id);
  res.json({ message: "Timetable deleted" });
});

export default router;
