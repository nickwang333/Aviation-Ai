import * as React from "react";
import * as Dialog from "@radix-ui/react-dialog";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/utils";

const modalVariants = cva(
  "fixed inset-0 flex items-center justify-center p-4 z-50 transition-all",
  {
    variants: {
      size: {
        sm: "max-w-sm",
        md: "max-w-md",
        lg: "max-w-lg",
        xl: "max-w-xl",
      },
      animation: {
        fade: "opacity-0 scale-95 data-[state=open]:opacity-100 data-[state=open]:scale-100",
        slide: "translate-y-4 opacity-0 data-[state=open]:translate-y-0 data-[state=open]:opacity-100",
      },
    },
    defaultVariants: {
      size: "md",
      animation: "fade",
    },
  }
);

export interface ModalProps
  extends React.ComponentPropsWithoutRef<typeof Dialog.Root>,
    VariantProps<typeof modalVariants> {
  title?: string;
  description?: string;
  className?: string;
  asChild?: boolean;
}

const Modal = React.forwardRef<HTMLDivElement, ModalProps>(
  ({ children, size, animation, title, description, className, asChild, ...props }, ref) => {
    return (
      <Dialog.Root {...props}>
        <Dialog.Portal>
          <Dialog.Overlay className="fixed inset-0 bg-black/50 backdrop-blur-sm data-[state=open]:animate-fadeIn" />
          <Dialog.Content
            ref={ref}
            className={cn(
              "bg-white dark:bg-gray-900 rounded-lg shadow-lg p-6",
              modalVariants({ size, animation }),
              className
            )}
          >
            {title && <Dialog.Title className="text-lg font-semibold">{title}</Dialog.Title>}
            {description && <Dialog.Description className="text-sm text-gray-500">{description}</Dialog.Description>}
            <div className="mt-4">{children}</div>
            <Dialog.Close className="absolute top-2 right-2 p-2 text-gray-400 hover:text-gray-600">
              ✖
            </Dialog.Close>
          </Dialog.Content>
        </Dialog.Portal>
      </Dialog.Root>
    );
  }
);

Modal.displayName = "Modal";

export { Modal };
