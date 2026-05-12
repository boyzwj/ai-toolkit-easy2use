import { NextResponse } from 'next/server';
import { PrismaClient } from '@prisma/client';

const prisma = new PrismaClient();
const TRAINING_DRAFT_KEY = 'TRAINING_DRAFT';

export async function GET() {
  try {
    const row = await prisma.settings.findUnique({
      where: { key: TRAINING_DRAFT_KEY },
    });

    if (!row?.value) {
      return NextResponse.json({ draft: null });
    }

    return NextResponse.json({ draft: JSON.parse(row.value) });
  } catch (error) {
    return NextResponse.json({ error: 'Failed to fetch training draft' }, { status: 500 });
  }
}

export async function POST(request: Request) {
  try {
    const body = await request.json();

    await prisma.settings.upsert({
      where: { key: TRAINING_DRAFT_KEY },
      update: { value: JSON.stringify(body) },
      create: { key: TRAINING_DRAFT_KEY, value: JSON.stringify(body) },
    });

    return NextResponse.json({ success: true });
  } catch (error) {
    return NextResponse.json({ error: 'Failed to save training draft' }, { status: 500 });
  }
}
