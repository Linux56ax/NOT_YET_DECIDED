import express from "express";
import RollingStock from "../models/RollingStock.js";

const router = express.Router();

// get all rolling stock
router.get("/", async (req, res) => {
  const stocks = await RollingStock.find().populate("train_id");
  res.json(stocks);
});

// get rolling stock by ID
router.get("/:id", async (req, res) => {
  const stock = await RollingStock.findById(req.params.id).populate("train_id");
  res.json(stock);
});

// post new rolling stock
router.post("/", async (req, res) => {
  const stock = new RollingStock(req.body);
  await stock.save();
  res.json(stock);
});

// put update rolling stock
router.put("/:id", async (req, res) => {
  const stock = await RollingStock.findByIdAndUpdate(req.params.id, req.body, { new: true });
  res.json(stock);
});

// delete rolling stock
router.delete("/:id", async (req, res) => {
  await RollingStock.findByIdAndDelete(req.params.id);
  res.json({ message: "Rolling stock deleted" });
});

export default router;
