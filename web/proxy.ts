import { clerkMiddleware } from "@clerk/nextjs/server";
import { NextResponse } from "next/server";

// Clerk keeps its session cookie fresh here. Without Clerk keys this does nothing.
export default process.env.NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY
  ? clerkMiddleware()
  : () => NextResponse.next();

export const config = {
  // Pages only: skip Next internals, static files, and /api (forwarded to the backend, which
  // checks the token itself).
  matcher: ["/((?!_next|api/|.*\\.[\\w]+$).*)"],
};
