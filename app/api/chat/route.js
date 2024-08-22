import { NextResponse } from "next/server";
import { Pinecone } from "@pinecone-database/pinecone";
import OpenAI from "openai";

const systemPrompt = `
You are a helpful and knowledgeable assistant that specializes in helping students find the best professors according to their 
specific needs and preferences. You use a combination of real-time retrieval of information and generated responses to provide 
accurate and useful recommendations. For each user query: 

Understanding User Preferences: You carefully analyze the user’s query to understand their specific requirements, such as the 
course subject, teaching style, difficulty level, availability of extra help, and any other preferences mentioned.

Information Retrieval: You access a database of professor ratings, reviews, and relevant course details to identify the top three 
professors that best match the user’s query. Consider factors such as overall rating, teaching effectiveness, clarity, availability, 
student reviews, and course compatibility.

Recommendation Generation: For each of the top three professors, provide a brief yet informative description including their name, 
department, rating, key strengths, notable student comments, and why they might be a good fit based on the user's query.

Tailoring Responses: If the user asks for more specific details, such as office hours or research interests, retrieve and provide 
that information in your response.

Balanced and Neutral: Maintain a neutral and balanced tone in your responses, presenting both the strengths and any potential 
considerations or challenges students might face with each professor.

Your goal is to help students make informed decisions by providing personalized and well-researched professor recommendations 
that align with their academic and personal preferences.
`

export async function POST(req) {
    const data = await req.json()
  
    const pc = new Pinecone({
      apiKey: process.env.PINECONE_API_KEY,
    })
    const index = pc.index('rag').namespace('ns1')
    const openai = new OpenAI()
  
    const text = data[data.length - 1].content
    const embedding = await openai.embeddings.create({
      model: 'text-embedding-3-small',
      input: text,
      encoding_format: 'float',
    })
  
    const results = await index.query({
      topK: 5,
      includeMetadata: true,
      vector: embedding.data[0].embedding,
    })
  
    let resultString = ''
    results.matches.forEach((match) => {
      resultString += `
      Returned Results:
      Professor: ${match.id}
      Review: ${match.metadata.review}
      Subject: ${match.metadata.subject}
      Stars: ${match.metadata.stars}
      \n\n`
    })
  
    const lastMessage = data[data.length - 1]
    const lastMessageContent = lastMessage.content + resultString
    const lastDataWithoutLastMessage = data.slice(0, data.length - 1)
  
    const completion = await openai.chat.completions.create({
      messages: [
        {role: 'system', content: systemPrompt},
        ...lastDataWithoutLastMessage,
        {role: 'user', content: lastMessageContent},
      ],
      model: 'gpt-3.5-turbo',
      stream: true,
    })
  
    const stream = new ReadableStream({
      async start(controller) {
        const encoder = new TextEncoder()
        try {
          for await (const chunk of completion) {
            const content = chunk.choices[0]?.delta?.content
            if (content) {
              const text = encoder.encode(content)
              controller.enqueue(text)
            }
          }
        } catch (err) {
          controller.error(err)
        } finally {
          controller.close()
        }
      },
    })
    return new NextResponse(stream)
  }