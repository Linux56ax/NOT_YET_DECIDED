import express from "express";
import Train from "../models/Train.js";

const router = express.Router();

// get all trains
router.get("/", async (req, res) => {
  const trains = await Train.find();
  res.json(trains);
});

// get train by ID
router.get("/:id", async (req, res) => {
  const train = await Train.findById(req.params.id);
  res.json(train);
});

// post new train
router.post("/", async (req, res) => {
  const train = new Train(req.body);
  await train.save();
  res.json(train);
});

// put update train
router.put("/:id", async (req, res) => {
  const train = await Train.findByIdAndUpdate(req.params.id, req.body, { new: true });
  res.json(train);
});

// delete train
router.delete("/:id", async (req, res) => {
  await Train.findByIdAndDelete(req.params.id);
  res.json({ message: "Train deleted" });
});

export default router;
