import { Html, Head, Main, NextScript } from 'next/document'

export default function Document() {
  return (
    <Html lang="en">
      <Head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <link 
          href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" 
          rel="stylesheet"
        />
        <meta name="description" content="DIY AutoML Platform for automatic machine learning model selection" />
        <meta name="color-scheme" content="light" />
      </Head>
      <body>
        <Main />
        <NextScript />
      </body>
    </Html>
  )
}
