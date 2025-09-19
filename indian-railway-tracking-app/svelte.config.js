import sveltePreprocess from 'svelte-preprocess';

const config = {
  preprocess: sveltePreprocess(),
  compilerOptions: {
    // enable run-time checks when not in production
    dev: !process.env.NODE_ENV === 'production',
  },
};

export default config;