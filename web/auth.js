// Accounts via Supabase Auth: passwordless email links plus Apple, Google and GitHub.
// The page is static, so this gate is a convenience, not security: the scanner code itself is public.

import { PROVIDERS, SUPABASE_ANON_KEY, SUPABASE_URL } from "./config.js";

export const authEnabled = Boolean(SUPABASE_URL && SUPABASE_ANON_KEY);

// The providers from PROVIDERS that are switched on in Supabase, so unfinished ones never show a broken button.
// If the settings can't be read, show them all and let Supabase report the problem on click.
export async function enabledProviders() {
  if (!authEnabled) return [];
  try {
    const res = await fetch(`${SUPABASE_URL}/auth/v1/settings`, { headers: { apikey: SUPABASE_ANON_KEY } });
    if (!res.ok) throw new Error(res.status);
    const { external = {} } = await res.json();
    return PROVIDERS.filter((p) => external[p]);
  } catch {
    return PROVIDERS;
  }
}

let client = null;

async function getClient() {
  if (!authEnabled) return null;
  if (!client) {
    const { createClient } = await import("./vendor/supabase.js");
    client = createClient(SUPABASE_URL, SUPABASE_ANON_KEY, {
      // Implicit flow so an email link still works when opened in a different browser or device.
      auth: { persistSession: true, autoRefreshToken: true, detectSessionInUrl: true, flowType: "implicit" },
    });
  }
  return client;
}

// Where Supabase sends people back after an email link or provider sign-in: this page, without any query or hash.
const returnUrl = () => location.origin + location.pathname;

export async function currentUser() {
  const sb = await getClient();
  if (!sb) return null;
  const { data } = await sb.auth.getSession();
  return data.session?.user ?? null;
}

export async function onAuthChange(callback) {
  const sb = await getClient();
  if (!sb) return;
  sb.auth.onAuthStateChange((_event, session) => callback(session?.user ?? null));
}

export async function sendEmailLink(email) {
  const sb = await getClient();
  const { error } = await sb.auth.signInWithOtp({ email, options: { emailRedirectTo: returnUrl(), shouldCreateUser: true } });
  if (error) throw error;
}

export async function signInWith(provider) {
  const sb = await getClient();
  const { error } = await sb.auth.signInWithOAuth({ provider, options: { redirectTo: returnUrl() } });
  if (error) throw error;
}

export async function signOut() {
  const sb = await getClient();
  await sb?.auth.signOut();
}
