// Which sign-in the app uses, fixed at build time:
//  - NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY set -> Clerk (production); run the backend with
//    EMOTEXT_AUTH_MODE=jwt and EMOTEXT_AUTH_JWKS_URL=https://<your clerk domain>/.well-known/jwks.json
//  - unset -> development sign-in (email only, no password), matching EMOTEXT_AUTH_MODE=dev
export const CLERK_ENABLED = Boolean(process.env.NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY);
