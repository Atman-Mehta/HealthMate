import mongoose from 'mongoose';

const otpSchema = new mongoose.Schema({
    email: { type: String, required: true },
    otp: { type: Number, required: true },
    createdAt: { type: Date, default: Date.now, expires: 300 }, // Expires after 5 minutes
});

const OTPModel = mongoose.model('OTP', otpSchema);

export default OTPModel;