# Setting up accounts

Accounts run on [Supabase Auth](https://supabase.com/auth) (free tier is enough). The project is `ai-scanner` (reference `peablbhvhwahogrscjio`), and its URL and publishable key are already in `web/config.js`.

Without an account people can paste text. With one they can also scan images, audio, video and documents. Because the site is static and the scanner runs in the browser, this gate is a convenience rather than real protection: someone determined could run the code without signing in.

The site address is `https://bigboy3900.github.io/ai-scanner/`.

## 1. Supabase project

Done: the project exists and `web/config.js` is filled in. One dashboard step remains:

1. **Authentication → URL Configuration**:
   - Site URL: `https://bigboy3900.github.io/ai-scanner/`
   - Redirect URLs: add `https://bigboy3900.github.io/ai-scanner/`

## 2. Email sign-in links (no password)

Email is on by default under **Authentication → Providers → Email**. Supabase's built-in email sender only allows a handful of emails per hour, which is fine for testing. For real use, add your own sender under **Authentication → Emails → SMTP Settings** (for example a free [Resend](https://resend.com) account).

## 3. Sign in with Google, GitHub and Apple

Each provider needs its own developer app. Every one of them asks for the same callback URL:

```
https://peablbhvhwahogrscjio.supabase.co/auth/v1/callback
```

After creating each app, paste its **Client ID** and **Client Secret** into **Authentication → Providers → (provider)** in Supabase and switch it on.

**GitHub** (free): github.com → Settings → Developer settings → OAuth Apps → New OAuth App. Homepage URL: the site address. Authorization callback URL: the callback above.

**Google** (free): console.cloud.google.com → APIs & Services. Set up the OAuth consent screen (External, app name, your email), then Credentials → Create credentials → OAuth client ID → Web application. Add the callback above under Authorized redirect URIs.

**Apple** (needs the paid Apple Developer Program, $99/year): in developer.apple.com → Certificates, Identifiers & Profiles, create an App ID with Sign in with Apple, a Services ID (this is the Client ID) with the callback above as its return URL and `peablbhvhwahogrscjio.supabase.co` as the domain, and a Sign in with Apple key (.p8). Supabase's Apple provider page explains how to turn the key into the client secret, which Apple makes expire every 6 months.

A button only appears on the site once its provider is switched on in Supabase, so there is nothing to change in the code.

## 4. Publish

Commit `web/config.js` and push to `main`. The site redeploys automatically.
